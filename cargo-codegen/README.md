# cargo-codegen

See the assembly your users get from your crate's hot paths.

- you write **probes**: small wrappers around your API, marked [`#[codegen::probe]`](../codegen-probe)
- `cargo codegen` compiles them **in a generated downstream crate**, so inlining matches your users'
- then it lists, counts and diffs their assembly
- mark functions [`#[codegen::watch]`](../codegen-probe) to see how many of each probe's instructions come from them

## Toy example

```toml
# Cargo.toml: the attributes are no-ops unless `cargo codegen` builds
[dependencies]
codegen-probe = { git = "https://github.com/fedemagnani/codegen" }
```

```rust
// src/lib.rs
#[codegen::watch]
pub fn get(values: &[u64], i: usize) -> u64 {
    values[i]
}

#[codegen::probe]
fn my_get(values: &[u64], i: usize) -> u64 {
    crate::get(values, i)
}
```

```sh
$ cargo install --git https://github.com/fedemagnani/codegen cargo-codegen
$ cargo codegen stats
== aarch64-apple-darwin
probe   instr  branch  jump  select  load  store  simd  barrier  call  panic
my_get     11       1     0       0     1      1     0        0     1      1
my_get: panics core::panicking::panic_bounds_check
watched        instr  branch  jump  select  load  store  simd  barrier  call  panic
my_get > get       8       1     0       0     1      0     0        0     1      1
```

## Commands

| Command | Does |
| --- | --- |
| `asm` | writes each probe's listing to `target/codegen/` |
| `stats` | counts branches, loads, stores, SIMD, barriers, calls, panics per probe and per watched function |
| `stats --mca` | adds [llvm-mca](https://llvm.org/docs/CommandGuide/llvm-mca.html)'s estimates on the host CPU, or `--mcpu <cpu>` |
| `save` | saves a baseline |
| `diff` | compares to the baseline, exit 1 on drift |
| `measure` | counts cycles, instructions, mispredicts and cache misses per call of each probe with inputs, on macOS |

Add `--target <triple>` for other targets. Run `cargo codegen --help` for every option.

## Reading `--mca`

llvm-mca runs each probe body as a straight-line loop: it ignores which branches are taken and models no mispredicts.

- `rthru`: cycles per call if only execution units limited it; the column that ranks probes best.
- `cycles`: simulated cycles per call, inflated when a return value feeds the next call's arguments.
- `deps%`, `press%`, `unit`: what stalled dispatch, and the busiest execution unit.
- `critical`: instructions on the longest dependency chain.

Needs `brew install llvm`, or `LLVM_MCA` set to the binary.

## Measuring

`measure` runs probes on inputs you write, counting what llvm-mca can't see: the branches that really mispredict, and
the loads that miss the cache.

```rust
#[codegen::probe]
mod probes {
    fn my_get(values: &[u64], i: usize) -> u64 {
        crate::get(values, i)
    }

    #[codegen::inputs(my_get)]
    fn my_get_inputs(m: &mut codegen::Measure) {
        let values: Vec<u64> = (0..1 << 20).collect();
        let calls: Vec<_> = m.rng().indices(1 << 16, values.len()).map(|i| (&values[..], i)).collect();
        m.run(&calls, |&(values, i)| my_get(values, i));
    }
}
```

- setup and the prepared inputs stay out of the counts; `m.case("hits")` names each distribution of the inputs
- `m.batched(setup, &calls, call)` rebuilds a state before every pass, for probes taking `&mut`
- `m.rng()` is seeded the same on every run
- each case: 11 samples of at least 100 000 calls, the median per call, less an empty loop's cost unless `--raw`
- the cost is per call among back-to-back independent calls, as in a loop: the core overlaps consecutive calls, so a
  call's latency can be several times its cost here, the way llvm-mca's `cycles` exceed its `rthru`
- `l1miss`: loads missing the L1 data cache; `l1wait`: cycles waiting on such a miss, which tells an L2 hit from DRAM;
  `tlbmiss`: loads and stores missing the L2 TLB. `-` where the CPU lacks the event, as M1 and M2 lack `l1wait`
- `ipc` always includes the loop: its instructions overlap the probe's, so subtracting its cycle would inflate ipc
- `spread%`: range of the samples' cycles over their median; a high one means a sample was disturbed
- a sample that ran on an efficiency core, even in part, is left out and reported; so is a case that ran only there
- the branch predictor memorizes inputs it sees again: random inputs replayed in each sample mispredict less than
  inputs seen once. 200 000 calls can still lose a fifth of their mispredicts. Each case therefore reports whether its
  mispredicts fell after the first pass, so you know when to give it more distinct calls
- the runner's probe code is checked against the listing, relocations aside, and a mismatch is reported
- counters come from Apple's kperf, so macOS only for now, as root: the runner runs under `sudo` unless `--no-sudo`, and without a
  terminal to type a password in, as in CI, only if sudo asks for none
- `--dry-run` runs each case once without counting, on any platform and without root
