//! `measure`: hardware counters per call of each probe with inputs, counted by a runner the host crate builds
//!
//! - an inputs function, `#[codegen::inputs(<probe>)]`, prepares a probe's calls; the runner counts them on the host
//! - before running, each measured probe's code in the runner is checked against its listing, relocations aside
//! - counters come from kperf, which needs root, so the runner runs under sudo unless already root

use std::fs::{self, File};
use std::path::Path;
use std::process::{Command, Stdio};

use object::{Object, ObjectSection, ObjectSymbol};

use crate::attribution::Inlines;
use crate::emit::{self, Build, Package};
use crate::listing::Listing;
use crate::target::Target;

/// Prefix of the runner's output lines carrying a row, as codegen-probe prints them
const ROW: &str = "codegen-measure";

/// Column headers of the table, after the probe and case
/// Counters per row: cycles, instructions, branches, mispredicts, L1 data cache load misses, cycles waiting on one,
/// L2 TLB misses
const COUNTERS: usize = 7;
const COLUMNS: [&str; COUNTERS + 2] = [
    "cycles", "instr", "branch", "mispred", "l1miss", "l1wait", "tlbmiss", "ipc", "spread%",
];

/// How `measure` runs
pub struct Options {
    /// Report the counted loops as they are, the empty loop's cost left in
    pub raw: bool,
    /// Run each case once without counting, which needs neither root nor macOS
    pub dry_run: bool,
    /// Run the runner under sudo when not root
    pub sudo: bool,
}

/// Builds the runner for the host, checks its probes against their listings, runs it and prints a row per case
pub fn run(package: &Package, triples: &[String], options: &Options) -> Result<(), String> {
    let host = emit::host_triple()?;
    if triples.iter().any(|triple| *triple != host) {
        return Err(format!(
            "measure runs the probes, so it builds for the host only, {host}"
        ));
    }
    if !options.dry_run && !cfg!(target_os = "macos") {
        return Err(
            "measure reads hardware counters through Apple's kperf, so it counts on macOS only for now; \
             --dry-run runs the inputs anywhere"
                .to_string(),
        );
    }
    let target = Target::parse(&host)?;
    let build = package.build(&target)?;
    let listing = Listing::parse(&build.assembly, &target);
    let probes = listing.probes()?;
    let measured: Vec<String> = listing.measured().into_iter().map(str::to_string).collect();
    if measured.is_empty() {
        return Err(format!(
            "no probe of {} has inputs: add a `#[codegen::inputs(<probe>)]` fn to its probe block",
            package.name
        ));
    }
    let runner = package.runner(&target, &build, &measured)?;
    let linked =
        fs::read(&runner).map_err(|error| format!("reading {}: {error}", runner.display()))?;
    let inlines = Inlines::parse(&build.object)?;
    let mut differing = Vec::new();
    for name in &measured {
        let probe = probes.iter().find(|probe| probe.name == name);
        let probe =
            probe.ok_or_else(|| format!("inputs of {name}, which the listing has no probe of"))?;
        // a merged probe's code is the body it shares
        let source = probe.merged_into.unwrap_or(probe.name);
        if !same_code(&inlines, &build, &linked, source, &target)? {
            differing.push(name);
        }
    }
    let output = execute(&runner, options)?;
    let mut rows = Vec::new();
    for line in output.lines() {
        match parse(line)? {
            Some(row) => rows.push(row),
            // what the inputs functions print themselves
            None => println!("{line}"),
        }
    }
    println!("== {}", target.triple);
    print!("{}", table(&rows, options.raw));
    for name in differing {
        println!(
            "{name}: the runner's code differs from the listing, so its counts may not describe it"
        );
    }
    let warnings = rows
        .iter()
        .flat_map(|row| [learned(row), on_efficiency(row)]);
    for warning in warnings.flatten() {
        println!("{warning}");
    }
    Ok(())
}

/// A warning when samples of the case ran on an efficiency core: left out, or all there is
fn on_efficiency(row: &Row) -> Option<String> {
    let counts = row.counts.as_ref()?;
    let (on_efficiency, samples) = (counts.on_efficiency, counts.samples);
    if on_efficiency == 0 {
        None
    } else if on_efficiency == samples {
        Some(format!(
            "{}: every sample ran on an efficiency core, so the counts describe one: measure again",
            label(row)
        ))
    } else {
        let left = if on_efficiency == 1 {
            "it is"
        } else {
            "they are"
        };
        Some(format!(
            "{}: {on_efficiency} of {samples} samples ran on an efficiency core, so {left} left out",
            label(row)
        ))
    }
}

/// Mispredicts per call the samples may fall below the first pass's before a case counts as learned
const LEARNED_DROP: f64 = 0.02;
/// Share of the first pass's mispredicts the samples may lose before a case counts as learned
const LEARNED_SHARE: f64 = 0.1;

/// A warning when the samples mispredict well below the first pass: the predictor memorized the replayed inputs,
/// so the counts describe those inputs, not inputs it sees once
fn learned(row: &Row) -> Option<String> {
    let counts = row.counts.as_ref()?;
    let (first, sampled) = (counts.first_mispredicts, counts.probe[3]);
    let drop = first - sampled;
    if drop < LEARNED_DROP || drop < first * LEARNED_SHARE {
        return None;
    }
    Some(format!(
        "{}: mispredicts fell from {first:.2} per call on the first pass to {sampled:.2}, \
         as the branch predictor learned the replayed inputs: give the case more distinct calls",
        label(row)
    ))
}

/// `probe/case`, or the probe alone for calls outside any case
fn label(row: &Row) -> String {
    if row.case.is_empty() {
        row.probe.clone()
    } else {
        format!("{}/{}", row.probe, row.case)
    }
}

/// A case of a probe as the runner reports it
#[derive(Debug, PartialEq)]
struct Row {
    probe: String,
    /// Empty for calls counted outside any named case
    case: String,
    calls: usize,
    /// None for a dry run
    counts: Option<Counts>,
}

/// Medians per call of cycles, instructions, branches and mispredicts
#[derive(Debug, PartialEq)]
struct Counts {
    /// NaN for a counter the CPU lacks
    probe: [f64; COUNTERS],
    empty_loop: [f64; COUNTERS],
    /// Range of the samples' cycles over their median
    spread: f64,
    /// Mispredicts per call of the first pass, before the predictor could learn the inputs
    first_mispredicts: f64,
    /// Samples that ran on an efficiency core, left out unless all did
    on_efficiency: usize,
    samples: usize,
}

/// The row a line of the runner's output carries, none for any other line
fn parse(line: &str) -> Result<Option<Row>, String> {
    let mut fields = line.split('\t');
    if fields.next() != Some(ROW) {
        return Ok(None);
    }
    let fields: Vec<&str> = fields.collect();
    let malformed = || format!("malformed line from the runner: {line:?}");
    let [probe, case, calls, rest @ ..] = fields.as_slice() else {
        return Err(malformed());
    };
    let calls = calls.parse().map_err(|_| malformed())?;
    let counts = match rest {
        ["dry"] => None,
        values => {
            let values: Vec<f64> = values
                .iter()
                // `-` for a counter the CPU lacks
                .map(|value| {
                    if *value == "-" {
                        Ok(f64::NAN)
                    } else {
                        value.parse()
                    }
                })
                .collect::<Result<_, _>>()
                .map_err(|_| malformed())?;
            let [
                probe @ ..,
                spread,
                first_mispredicts,
                on_efficiency,
                samples,
            ] = values.as_slice()
            else {
                return Err(malformed());
            };
            let (probe, empty_loop) = probe.split_at_checked(COUNTERS).ok_or_else(malformed)?;
            Some(Counts {
                probe: probe.try_into().map_err(|_| malformed())?,
                empty_loop: empty_loop.try_into().map_err(|_| malformed())?,
                spread: *spread,
                first_mispredicts: *first_mispredicts,
                on_efficiency: *on_efficiency as usize,
                samples: *samples as usize,
            })
        }
    };
    Ok(Some(Row {
        probe: probe.to_string(),
        case: case.to_string(),
        calls,
        counts,
    }))
}

/// A row per case, `probe/case` or the probe alone, counts per call less the empty loop's unless `raw`, ipc always raw
fn table(rows: &[Row], raw: bool) -> String {
    let width = rows.iter().map(|row| label(row).len()).max().unwrap_or(0);
    let width = width.max("probe".len());
    let header: Vec<String> = COLUMNS
        .iter()
        .map(|column| format!("{column:>8}"))
        .collect();
    let mut out = format!("{:<width$} {}\n", "probe", header.join(""));
    for row in rows {
        let Some(counts) = &row.counts else {
            out += &format!(
                "{:<width$}   dry run: {} calls ran once, uncounted\n",
                label(row),
                row.calls
            );
            continue;
        };
        let per_call: [f64; COUNTERS] = if raw {
            counts.probe
        } else {
            std::array::from_fn(|i| counts.probe[i] - counts.empty_loop[i])
        };
        // from the raw counts: the loop's instructions overlap the probe's, so subtracting its whole cycle inflates ipc
        let [cycles, instructions, ..] = counts.probe;
        let mut cells: Vec<String> = per_call
            .iter()
            .map(|value| {
                if value.is_nan() {
                    format!("{:>8}", "-")
                } else {
                    format!("{value:>8.2}")
                }
            })
            .collect();
        let ipc = if cycles > 0.0 {
            format!("{:>8.2}", instructions / cycles)
        } else {
            format!("{:>8}", "-")
        };
        cells.push(ipc);
        cells.push(format!("{:>8.1}", counts.spread * 100.0));
        out += &format!("{:<width$} {}\n", label(row), cells.join(""));
    }
    out
}

/// Whether the runner holds `symbol`'s code as listed: the same bytes, those relocations patch aside
fn same_code(
    inlines: &Inlines<'_>,
    build: &Build,
    runner: &[u8],
    symbol: &str,
    target: &Target,
) -> Result<bool, String> {
    let extent = inlines.extent(symbol)?;
    let name = format!("{}{symbol}", target.symbol_prefix());
    let object = object::File::parse(&*build.object)
        .map_err(|error| format!("parsing the object: {error}"))?;
    let linked =
        object::File::parse(runner).map_err(|error| format!("parsing the runner: {error}"))?;
    let (listed, relocated) = code(&object, &name, extent)?;
    let (ran, _) = code(&linked, &name, extent)?;
    Ok(masked_equal(&listed, &ran, &relocated))
}

/// The `extent` bytes from symbol `name`, and the offsets into them where relocations apply
fn code(file: &object::File<'_>, name: &str, extent: u64) -> Result<(Vec<u8>, Vec<usize>), String> {
    let symbol = file.symbols().find(|symbol| symbol.name() == Ok(name));
    let symbol = symbol.ok_or_else(|| format!("no symbol {name}"))?;
    let index = symbol
        .section_index()
        .ok_or_else(|| format!("symbol {name} is in no section"))?;
    let section = file
        .section_by_index(index)
        .map_err(|error| format!("reading {name}'s section: {error}"))?;
    let data = section
        .data()
        .map_err(|error| format!("reading {name}'s section: {error}"))?;
    let start = symbol.address() - section.address();
    let end = start + extent;
    let bytes = usize::try_from(start)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(start, end)| data.get(start..end));
    let bytes = bytes.ok_or_else(|| format!("{name} runs past its section"))?;
    let relocated = section
        .relocations()
        .filter(|(offset, _)| (start..end).contains(offset))
        .filter_map(|(offset, _)| usize::try_from(offset - start).ok())
        .collect();
    Ok((bytes.to_vec(), relocated))
}

/// Whether `a` and `b` are equal outside the 4 bytes from each offset in `relocated`, the width of the fields patched
fn masked_equal(a: &[u8], b: &[u8], relocated: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let patched = |i: usize| {
        relocated
            .iter()
            .any(|&offset| (offset..offset + 4).contains(&i))
    };
    a.iter()
        .zip(b)
        .enumerate()
        .all(|(i, (a, b))| a == b || patched(i))
}

/// Runs the runner, under sudo when counting needs root, returning its standard output
fn execute(runner: &Path, options: &Options) -> Result<String, String> {
    let elevate = !options.dry_run && options.sudo && !is_root();
    // sudo asks for a password on the terminal; without one, as in CI, it may only run without asking
    let prompts = File::open("/dev/tty").is_ok();
    let mut command = if elevate {
        eprintln!(
            "cargo-codegen: reading hardware counters needs root, so the runner runs under sudo"
        );
        let mut sudo = Command::new("sudo");
        if !prompts {
            sudo.arg("-n");
        }
        sudo.arg("--").arg(runner);
        sudo
    } else {
        Command::new(runner)
    };
    if options.dry_run {
        command.arg("--dry-run");
    }
    // sudo's prompt and the runner's errors reach the terminal
    command.stdin(Stdio::inherit()).stderr(Stdio::inherit());
    let output = command
        .output()
        .map_err(|error| format!("running {}: {error}", runner.display()))?;
    // sudo exits 1 when it can't run the command, the runner 2 on its own errors
    if elevate && !prompts && output.status.code() == Some(1) {
        return Err(
            "sudo needs a password, and there is no terminal to ask for it: run as root, allow the runner in \
             sudoers without a password, or count nothing with --dry-run"
                .to_string(),
        );
    }
    if !output.status.success() {
        return Err(format!(
            "the runner {} failed, {}",
            runner.display(),
            output.status
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(unix)]
fn is_root() -> bool {
    unsafe extern "C" {
        safe fn geteuid() -> u32;
    }
    geteuid() == 0
}

#[cfg(not(unix))]
fn is_root() -> bool {
    false
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rows_are_parsed_from_the_runner_lines_only() {
        let counted = "codegen-measure\tget\thits\t200000\t20.0\t29.0\t5.5\t0.5\t1.0\t9.0\t-\t1.0\t3.0\t1.0\t0.0\t0.0\t0.0\t-\t0.04\t0.52\t2\t11";
        let dry = "codegen-measure\tget\t\t16\tdry";
        let row = parse(counted).unwrap().unwrap();
        assert_eq!(row.probe, "get");
        assert_eq!(row.case, "hits");
        assert_eq!(row.calls, 200_000);
        let counts = row.counts.unwrap();
        assert_eq!(counts.probe[..6], [20.0, 29.0, 5.5, 0.5, 1.0, 9.0]);
        assert_eq!(counts.empty_loop[..6], [1.0, 3.0, 1.0, 0.0, 0.0, 0.0]);
        // a counter the CPU lacks
        assert!(counts.probe[6].is_nan() && counts.empty_loop[6].is_nan());
        assert_eq!(counts.first_mispredicts, 0.52);
        assert_eq!((counts.on_efficiency, counts.samples), (2, 11));
        let dry = parse(dry).unwrap().unwrap();
        assert_eq!(dry.case, "");
        assert_eq!(dry.counts, None);
        assert_eq!(parse("printed by the inputs").unwrap(), None);
        assert!(parse("codegen-measure\tget\thits\tmany\tdry").is_err());
    }

    #[test]
    fn the_table_subtracts_the_empty_loop_unless_raw() {
        let rows = [Row {
            probe: "get".to_string(),
            case: "mixed".to_string(),
            calls: 10,
            counts: Some(Counts {
                probe: [21.0, 33.0, 6.5, 0.5, 1.0, 9.0, f64::NAN],
                empty_loop: [1.0, 3.0, 1.0, 0.0, 0.0, 0.0, f64::NAN],
                spread: 0.05,
                first_mispredicts: 0.5,
                on_efficiency: 0,
                samples: 11,
            }),
        }];
        let subtracted = table(&rows, false);
        let raw = table(&rows, true);
        let line = |text: &str| {
            text.lines()
                .nth(1)
                .unwrap()
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            line(&subtracted),
            [
                "get/mixed",
                "20.00",
                "30.00",
                "5.50",
                "0.50",
                "1.00",
                "9.00",
                "-",
                "1.57",
                "5.0"
            ]
        );
        assert_eq!(line(&raw)[1], "21.00");
        // ipc counts the loop in both: 33 / 21, not 30 / 20
        assert_eq!(line(&raw)[8], "1.57");
    }

    #[test]
    fn a_case_whose_mispredicts_fall_after_the_first_pass_is_learned() {
        let row = |first_mispredicts, sampled| Row {
            probe: "get".to_string(),
            case: "mixed".to_string(),
            calls: 20_000,
            counts: Some(Counts {
                probe: [6.0, 20.0, 3.9, sampled, 0.0, 0.0, 0.0],
                empty_loop: [1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                spread: 0.05,
                first_mispredicts,
                on_efficiency: 0,
                samples: 11,
            }),
        };
        // letters/mixed replaying 20 000 keys, then 2 000 000
        let warning = learned(&row(0.44, 0.01)).unwrap();
        assert!(
            warning.starts_with("get/mixed: mispredicts fell from 0.44"),
            "{warning}"
        );
        assert_eq!(learned(&row(0.44, 0.44)), None);
        // a cold predictor's few mispredicts on a predictable case
        assert_eq!(learned(&row(0.003, 0.0)), None);
    }

    #[test]
    fn samples_on_an_efficiency_core_are_reported() {
        let row = |on_efficiency| Row {
            probe: "get".to_string(),
            case: "hits".to_string(),
            calls: 1000,
            counts: Some(Counts {
                probe: [6.0, 20.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                empty_loop: [1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                spread: 0.01,
                first_mispredicts: 0.0,
                on_efficiency,
                samples: 11,
            }),
        };
        assert_eq!(on_efficiency(&row(0)), None);
        let some = on_efficiency(&row(3)).unwrap();
        assert!(some.starts_with("get/hits: 3 of 11 samples"), "{some}");
        let one = on_efficiency(&row(1)).unwrap();
        assert!(
            one.ends_with("1 of 11 samples ran on an efficiency core, so it is left out"),
            "{one}"
        );
        let all = on_efficiency(&row(11)).unwrap();
        assert!(all.starts_with("get/hits: every sample"), "{all}");
    }

    #[test]
    fn relocated_fields_are_ignored() {
        let listed = [1, 2, 3, 4, 0, 0, 0, 0, 9];
        let linked = [1, 2, 3, 4, 7, 7, 7, 7, 9];
        assert!(masked_equal(&listed, &linked, &[4]));
        assert!(!masked_equal(&listed, &linked, &[]));
        assert!(!masked_equal(&listed, &linked[..8], &[4]));
    }
}
