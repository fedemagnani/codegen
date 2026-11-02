//! Hardware counters per probe call, over the inputs a crate gives its probes, for `cargo codegen measure`
//!
//! - an inputs function, `#[codegen::inputs(<probe>)]` in a probe block, prepares calls and runs them through [`Measure`]
//! - only the calls are counted: setup, preparing the inputs and dropping them happen outside the counted loops
//! - each case is counted over [`SAMPLES`] samples of at least [`MIN_CALLS`] calls, the median of each counter kept
//! - an empty loop over the same inputs is counted too, so `cargo codegen measure` can subtract the loop's own cost
//! - a sample that ran on an efficiency core, even in part, is left out and reported: passes are counted in chunks of
//!   [`MIN_CALLS`] calls, the core checked between chunks
//! - the first pass's mispredicts are reported too, so `cargo codegen measure` can tell when the predictor learned
//!   the inputs by replaying them

use std::hint::black_box;
use std::process::ExitCode;

#[cfg(target_os = "macos")]
use crate::kperf::Kperf;

/// Samples counted per case, the median of each counter reported
const SAMPLES: usize = 11;
/// Calls a sample makes at least, short inputs passed over several times so the counter reads' own cost vanishes;
/// long ones are counted in chunks of as many calls
const MIN_CALLS: usize = 100_000;
/// Seed every probe's [`Rng`] starts from, so its inputs are the same on every run
const SEED: u64 = 0x5EED_C0DE_6E4E_5EED;
/// Prefix of the lines `cargo codegen measure` reads from the runner's output
const ROW: &str = "codegen-measure";

/// Counters read: cycles, instructions, branches, mispredicted branches, loads missing the L1 data cache, cycles waiting
/// on such a miss, and loads and stores missing the L2 TLB
pub(crate) const COUNTERS: usize = 7;
pub(crate) type Counts = [u64; COUNTERS];

/// The counters an inputs function runs its probe's calls through, one case at a time
///
/// ```rust,ignore
/// #[codegen::probe]
/// mod probes {
///     fn my_get(values: &[u64], i: usize) -> u64 {
///         crate::get(values, i)
///     }
///
///     #[codegen::inputs(my_get)]
///     fn my_get_inputs(m: &mut codegen::Measure) {
///         let values: Vec<u64> = (0..1 << 20).collect();
///         let calls: Vec<_> = m.rng().indices(1 << 16, values.len()).map(|i| (&values[..], i)).collect();
///         m.run(&calls, |&(values, i)| my_get(values, i));
///     }
/// }
/// ```
pub struct Measure {
    counters: Counters,
    probe: &'static str,
    rng: Rng,
}

/// Where counts come from
enum Counters {
    #[cfg(target_os = "macos")]
    Kperf(Box<Kperf>),
    /// `--dry-run`: each case runs once, nothing is counted
    Dry,
}

impl Measure {
    /// A measure counting nothing, for tests calling inputs functions directly
    #[cfg(test)]
    pub(crate) fn dry() -> Self {
        Self {
            counters: Counters::Dry,
            probe: "",
            rng: Rng::new(SEED),
        }
    }

    /// Runs every inputs function, as the runner `cargo codegen measure` generates does; `--dry-run` counts nothing
    #[doc(hidden)]
    pub fn __main(inputs: &[fn(&mut Measure)]) -> ExitCode {
        let dry = std::env::args().any(|arg| arg == "--dry-run");
        let counters = match Counters::start(dry) {
            Ok(counters) => counters,
            Err(message) => {
                eprintln!("{message}");
                return ExitCode::from(2);
            }
        };
        let mut measure = Self {
            counters,
            probe: "",
            rng: Rng::new(SEED),
        };
        for run in inputs {
            run(&mut measure);
        }
        ExitCode::SUCCESS
    }

    /// Names the probe the next cases count, restarting the generator so its inputs don't depend on the others
    #[doc(hidden)]
    pub fn __probe(&mut self, probe: &'static str) {
        self.probe = probe;
        self.rng = Rng::new(SEED);
    }

    /// A generator seeded the same for every probe on every run, for inputs that are random but reproducible
    pub fn rng(&mut self) -> &mut Rng {
        &mut self.rng
    }

    /// A case named `name`, telling apart distributions of the same probe's inputs, such as hits and misses
    pub fn case(&mut self, name: &str) -> Case<'_> {
        Case {
            measure: self,
            name: name.to_string(),
        }
    }

    /// Counts one `call` per element of `calls`; the call's output is dropped inside the counted loop
    pub fn run<A, R>(&mut self, calls: &[A], call: impl FnMut(&A) -> R) {
        self.case("").run(calls, call);
    }

    /// Counts `call` on each element of `calls` against a state `setup` rebuilds, uncounted, before every pass
    pub fn batched<S, A, R>(
        &mut self,
        setup: impl FnMut() -> S,
        calls: &[A],
        call: impl FnMut(&mut S, &A) -> R,
    ) {
        self.case("").batched(setup, calls, call);
    }

    /// Counts `SAMPLES` samples of `passes` passes of `pass` and of `empty`, printing the medians per call
    fn count(
        &mut self,
        case: &str,
        calls: usize,
        passes: usize,
        mut pass: impl FnMut(&Counters) -> Pass,
        mut empty: impl FnMut(&Counters) -> Pass,
    ) {
        assert!(
            calls > 0,
            "case {case:?} of probe {} has no calls",
            self.probe
        );
        if !self.counters.counting() {
            pass(&self.counters);
            println!("{ROW}\t{}\t{case}\t{calls}\tdry", self.probe);
            return;
        }
        // warmed up, so caches and predictors start as they are in the samples after; the predictor sees the
        // inputs first here, so its mispredicts are those of inputs it hasn't learned
        let first = pass(&self.counters).counts;
        empty(&self.counters);
        let mut probe_samples = Vec::with_capacity(SAMPLES);
        let mut loop_samples = Vec::with_capacity(SAMPLES);
        let mut efficiency = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let mut probe = [0; COUNTERS];
            let mut empty_loop = [0; COUNTERS];
            let mut on_efficiency = false;
            for _ in 0..passes {
                let (probe_pass, empty_pass) = (pass(&self.counters), empty(&self.counters));
                add(&mut probe, probe_pass.counts);
                add(&mut empty_loop, empty_pass.counts);
                // the probe on a performance core and the loop on an efficiency one skew the subtraction too
                on_efficiency |= probe_pass.on_efficiency || empty_pass.on_efficiency;
            }
            probe_samples.push(probe);
            loop_samples.push(empty_loop);
            efficiency.push(on_efficiency);
        }
        let on_efficiency = efficiency.iter().filter(|&&on| on).count();
        let (probe_samples, loop_samples) = performance(&efficiency, probe_samples, loop_samples);
        let total = (calls * passes) as f64;
        let probe = medians(&probe_samples).map(|count| count as f64 / total);
        let empty_loop = medians(&loop_samples).map(|count| count as f64 / total);
        let spread = spread(&probe_samples);
        let first_mispredicts = first[3] as f64 / calls as f64;
        let available = self.counters.available();
        let cells: Vec<String> = probe
            .iter()
            .chain(&empty_loop)
            .enumerate()
            .map(|(i, value)| {
                if available[i % COUNTERS] {
                    format!("{value:.4}")
                } else {
                    "-".to_string()
                }
            })
            .collect();
        println!(
            "{ROW}\t{}\t{case}\t{calls}\t{}\t{spread:.4}\t{first_mispredicts:.4}\t{on_efficiency}\t{SAMPLES}",
            self.probe,
            cells.join("\t")
        );
    }
}

/// A named distribution of a probe's inputs, counted by [`Case::run`] or [`Case::batched`]
pub struct Case<'m> {
    measure: &'m mut Measure,
    name: String,
}

impl Case<'_> {
    /// [`Measure::run`] under this case's name
    pub fn run<A, R>(self, calls: &[A], mut call: impl FnMut(&A) -> R) {
        let passes = MIN_CALLS.div_ceil(calls.len().max(1));
        let pass = |counters: &Counters| {
            chunked(counters, calls, |input| {
                black_box(call(input));
            })
        };
        let empty = |counters: &Counters| empty_pass(counters, calls);
        self.measure
            .count(&self.name, calls.len(), passes, pass, empty);
    }

    /// [`Measure::batched`] under this case's name
    pub fn batched<S, A, R>(
        self,
        mut setup: impl FnMut() -> S,
        calls: &[A],
        mut call: impl FnMut(&mut S, &A) -> R,
    ) {
        let passes = MIN_CALLS.div_ceil(calls.len().max(1));
        let pass = |counters: &Counters| {
            let mut state = setup();
            let pass = chunked(counters, calls, |input| {
                black_box(call(&mut state, input));
            });
            // dropped after the last read, like the setup done before the first
            drop(state);
            pass
        };
        let empty = |counters: &Counters| empty_pass(counters, calls);
        self.measure
            .count(&self.name, calls.len(), passes, pass, empty);
    }
}

impl Counters {
    /// The counters of this thread, or none for a dry run
    fn start(dry: bool) -> Result<Self, String> {
        if dry {
            return Ok(Self::Dry);
        }
        #[cfg(target_os = "macos")]
        {
            Kperf::start().map(|kperf| Self::Kperf(Box::new(kperf)))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(
                "reading hardware counters needs macOS for now, where kperf provides them"
                    .to_string(),
            )
        }
    }

    /// Whether there are counters to read, not for a dry run; a match, as `Dry` is the only variant without a backend
    fn counting(&self) -> bool {
        match self {
            #[cfg(target_os = "macos")]
            Self::Kperf(_) => true,
            Self::Dry => false,
        }
    }

    /// Whether the thread runs on an efficiency core, never for a dry run
    fn on_efficiency_core(&self) -> bool {
        match self {
            #[cfg(target_os = "macos")]
            Self::Kperf(kperf) => kperf.on_efficiency_core(),
            Self::Dry => false,
        }
    }

    /// Which counters the CPU has, the others read as zero
    fn available(&self) -> [bool; COUNTERS] {
        match self {
            #[cfg(target_os = "macos")]
            Self::Kperf(kperf) => kperf.available(),
            Self::Dry => [true; COUNTERS],
        }
    }

    /// This thread's running counts, zeros for a dry run
    fn read(&self) -> Counts {
        match self {
            #[cfg(target_os = "macos")]
            Self::Kperf(kperf) => kperf.read(),
            Self::Dry => [0; COUNTERS],
        }
    }
}

/// Counts of one pass over some calls, and whether any of it ran on an efficiency core
struct Pass {
    counts: Counts,
    on_efficiency: bool,
}

/// One pass of `call` over `calls`, counted [`MIN_CALLS`] calls at a time, the core checked before the first chunk
/// and after each, outside the counts
fn chunked<A>(counters: &Counters, calls: &[A], mut call: impl FnMut(&A)) -> Pass {
    let mut pass = Pass {
        counts: [0; COUNTERS],
        on_efficiency: counters.on_efficiency_core(),
    };
    for chunk in calls.chunks(MIN_CALLS) {
        let before = counters.read();
        for input in chunk {
            call(input);
        }
        add(&mut pass.counts, sub(counters.read(), before));
        pass.on_efficiency |= counters.on_efficiency_core();
    }
    pass
}

/// One pass over `calls` doing nothing but hand each input to `black_box`, as the counted loops do outputs
fn empty_pass<A>(counters: &Counters, calls: &[A]) -> Pass {
    chunked(counters, calls, |input| {
        black_box(input);
    })
}

fn add(total: &mut Counts, counts: Counts) {
    for (total, count) in total.iter_mut().zip(counts) {
        *total += count;
    }
}

fn sub(after: Counts, before: Counts) -> Counts {
    std::array::from_fn(|i| after[i].wrapping_sub(before[i]))
}

/// Median of each counter over the samples
fn medians(samples: &[Counts]) -> Counts {
    std::array::from_fn(|i| {
        let mut values: Vec<u64> = samples.iter().map(|sample| sample[i]).collect();
        values.sort_unstable();
        values[values.len() / 2]
    })
}

/// The probe and loop samples that ran on performance cores only, or all of them if none did
fn performance(
    efficiency: &[bool],
    probe: Vec<Counts>,
    empty_loop: Vec<Counts>,
) -> (Vec<Counts>, Vec<Counts>) {
    if efficiency.iter().all(|&on| on) {
        return (probe, empty_loop);
    }
    let keep = |samples: Vec<Counts>| -> Vec<Counts> {
        samples
            .into_iter()
            .zip(efficiency)
            .filter_map(|(sample, &on)| (!on).then_some(sample))
            .collect()
    };
    (keep(probe), keep(empty_loop))
}

/// Range of the samples' cycles over their median, high when a sample was disturbed
fn spread(samples: &[Counts]) -> f64 {
    let cycles: Vec<u64> = samples.iter().map(|sample| sample[0]).collect();
    let min = cycles.iter().min().copied().unwrap_or(0);
    let max = cycles.iter().max().copied().unwrap_or(0);
    let median = medians(samples)[0];
    if median == 0 {
        return 0.0;
    }
    (max - min) as f64 / median as f64
}

/// SplitMix64: random enough for inputs, and the same sequence from the same seed
pub struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number below `n`, uniform enough for any `n` an input set has
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// `count` numbers below `n`, such as indices into a table of `n` entries
    pub fn indices(&mut self, count: usize, n: usize) -> impl Iterator<Item = usize> + '_ {
        (0..count).map(move |_| self.below(n))
    }

    /// Shuffles `items` in place, so short repeating sequences don't train the branch predictor
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn the_generator_repeats_from_its_seed() {
        let first: Vec<u64> = (0..4)
            .map({
                let mut rng = Rng::new(SEED);
                move |_| rng.next_u64()
            })
            .collect();
        let mut again = Rng::new(SEED);
        let second: Vec<u64> = (0..4).map(|_| again.next_u64()).collect();
        assert_eq!(first, second);
        assert_ne!(first[0], first[1]);
    }

    #[test]
    fn numbers_stay_below_their_bound() {
        let mut rng = Rng::new(SEED);
        let below = rng.indices(1000, 7).all(|i| i < 7);
        assert!(below);
        let mut items: Vec<u32> = (0..100).collect();
        rng.shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_ne!(items, sorted);
        assert_eq!(sorted, (0..100).collect::<Vec<u32>>());
    }

    #[test]
    fn a_pass_calls_once_per_input_across_chunks() {
        let calls: Vec<usize> = (0..MIN_CALLS * 2 + 7).collect();
        let mut seen = 0;
        let pass = chunked(&Counters::Dry, &calls, |&input| {
            assert_eq!(input, seen);
            seen += 1;
        });
        assert_eq!(seen, calls.len());
        assert!(!pass.on_efficiency);
    }

    #[test]
    fn samples_on_an_efficiency_core_are_left_out_unless_all_are() {
        let probe = vec![
            [6, 1, 1, 0, 0, 0, 0],
            [10, 1, 1, 0, 0, 0, 0],
            [6, 1, 1, 0, 0, 0, 0],
        ];
        let empty_loop = vec![
            [1, 1, 1, 0, 0, 0, 0],
            [2, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ];
        let (kept, kept_loop) =
            performance(&[false, true, false], probe.clone(), empty_loop.clone());
        assert_eq!(kept, [[6, 1, 1, 0, 0, 0, 0], [6, 1, 1, 0, 0, 0, 0]]);
        assert_eq!(kept_loop, [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]]);
        let (all, _) = performance(&[true, true, true], probe.clone(), empty_loop);
        assert_eq!(all, probe);
    }

    #[test]
    fn medians_and_spread_ignore_one_disturbed_sample() {
        let samples = [
            [100, 1, 1, 0, 0, 0, 0],
            [102, 1, 1, 0, 0, 0, 0],
            [300, 1, 1, 0, 0, 0, 0],
            [98, 1, 1, 0, 0, 0, 0],
            [101, 1, 1, 0, 0, 0, 0],
        ];
        assert_eq!(medians(&samples), [101, 1, 1, 0, 0, 0, 0]);
        let spread = spread(&samples);
        assert!((spread - 202.0 / 101.0).abs() < 1e-9, "{spread}");
    }
}
