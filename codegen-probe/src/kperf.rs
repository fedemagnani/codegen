//! This thread's hardware counters through Apple's private kperf frameworks, which need root
//!
//! - events are looked up by name in whatever PMC database the CPU loads, so new chips need no table of known CPUs
//! - the frameworks are opened with `dlopen`, their functions resolved by name, as Instruments' own clients do
//! - the efficiency cores are the lowest-numbered CPUs, `hw.perflevel1.logicalcpu` of them, as on the M4 Pro

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

use crate::measure::{COUNTERS, Counts};

/// Events by their name in Apple's PMC databases, in [`Counts`] order
const EVENTS: [&CStr; COUNTERS] = [
    c"FIXED_CYCLES",
    c"FIXED_INSTRUCTIONS",
    c"INST_BRANCH",
    c"BRANCH_MISPRED_NONSPEC",
    c"L1D_CACHE_MISS_LD_NONSPEC",
    c"LDST_UNIT_OLD_L1D_CACHE_MISS",
    c"L2_TLB_MISS_DATA",
];
/// Events every generation has; the others are left out where the database lacks them, as M1 and M2 lack
/// `LDST_UNIT_OLD_L1D_CACHE_MISS`
const REQUIRED: usize = 4;
/// Counters a thread has at most
const MAX_COUNTERS: usize = 32;
/// Class of the configurable counters, which the branch events need
const CLASS_CONFIGURABLE_MASK: u32 = 1 << 1;
/// `QOS_CLASS_USER_INTERACTIVE`, steering the thread to performance cores
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
const RTLD_NOW: c_int = 0x2;

unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
    fn pthread_cpu_number_np(cpu: *mut usize) -> c_int;
    fn sysctlbyname(
        name: *const c_char,
        old: *mut c_void,
        old_len: *mut usize,
        new: *mut c_void,
        new_len: usize,
    ) -> c_int;
}

/// Opaque `kpep_db`
enum Db {}
/// Opaque `kpep_config`
enum Config {}
/// Opaque `kpep_event`
enum Event {}

/// Functions of `kperf.framework`
struct Kpc {
    force_all_ctrs_get: unsafe extern "C" fn(*mut c_int) -> c_int,
    force_all_ctrs_set: unsafe extern "C" fn(c_int) -> c_int,
    set_config: unsafe extern "C" fn(u32, *mut u64) -> c_int,
    set_counting: unsafe extern "C" fn(u32) -> c_int,
    set_thread_counting: unsafe extern "C" fn(u32) -> c_int,
    get_thread_counters: unsafe extern "C" fn(u32, u32, *mut u64) -> c_int,
}

/// Functions of `kperfdata.framework`
struct Kpep {
    db_create: unsafe extern "C" fn(*const c_char, *mut *mut Db) -> c_int,
    db_free: unsafe extern "C" fn(*mut Db),
    db_event: unsafe extern "C" fn(*mut Db, *const c_char, *mut *mut Event) -> c_int,
    config_create: unsafe extern "C" fn(*mut Db, *mut *mut Config) -> c_int,
    config_free: unsafe extern "C" fn(*mut Config),
    config_force_counters: unsafe extern "C" fn(*mut Config) -> c_int,
    config_add_event: unsafe extern "C" fn(*mut Config, *mut *mut Event, u32, *mut u32) -> c_int,
    config_kpc_classes: unsafe extern "C" fn(*mut Config, *mut u32) -> c_int,
    config_kpc_map: unsafe extern "C" fn(*mut Config, *mut usize, usize) -> c_int,
    config_kpc: unsafe extern "C" fn(*mut Config, *mut u64, usize) -> c_int,
}

/// The calling thread's counters, programmed for [`EVENTS`]; dropping it hands them back to the system
pub struct Kperf {
    kpc: Kpc,
    kpep: Kpep,
    db: *mut Db,
    config: *mut Config,
    /// Where each event's count lands in the thread's counter array, none for an event the CPU lacks
    slots: [Option<usize>; COUNTERS],
    /// The system's `force_all_ctrs` before the counters were taken
    saved_force_all: c_int,
    /// CPUs numbered below this are efficiency cores
    efficiency_cpus: usize,
}

impl Kperf {
    /// Loads the frameworks, takes the counters from the system and starts counting on this thread
    pub fn start() -> Result<Self, String> {
        let kperf = open(c"/System/Library/PrivateFrameworks/kperf.framework/kperf")?;
        let kperfdata = open(c"/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata")?;
        // SAFETY: each symbol is resolved by its name in the framework that exports it, with its C signature
        let (kpc, kpep) = unsafe {
            let kpc = Kpc {
                force_all_ctrs_get: symbol(kperf, c"kpc_force_all_ctrs_get")?,
                force_all_ctrs_set: symbol(kperf, c"kpc_force_all_ctrs_set")?,
                set_config: symbol(kperf, c"kpc_set_config")?,
                set_counting: symbol(kperf, c"kpc_set_counting")?,
                set_thread_counting: symbol(kperf, c"kpc_set_thread_counting")?,
                get_thread_counters: symbol(kperf, c"kpc_get_thread_counters")?,
            };
            let kpep = Kpep {
                db_create: symbol(kperfdata, c"kpep_db_create")?,
                db_free: symbol(kperfdata, c"kpep_db_free")?,
                db_event: symbol(kperfdata, c"kpep_db_event")?,
                config_create: symbol(kperfdata, c"kpep_config_create")?,
                config_free: symbol(kperfdata, c"kpep_config_free")?,
                config_force_counters: symbol(kperfdata, c"kpep_config_force_counters")?,
                config_add_event: symbol(kperfdata, c"kpep_config_add_event")?,
                config_kpc_classes: symbol(kperfdata, c"kpep_config_kpc_classes")?,
                config_kpc_map: symbol(kperfdata, c"kpep_config_kpc_map")?,
                config_kpc: symbol(kperfdata, c"kpep_config_kpc")?,
            };
            (kpc, kpep)
        };
        let mut saved_force_all = 0;
        // SAFETY: reads a sysctl into a local; it fails without root
        let status = unsafe { (kpc.force_all_ctrs_get)(&raw mut saved_force_all) };
        if status != 0 {
            return Err(
                "reading hardware counters needs root: run `cargo codegen measure` without --no-sudo, or as root"
                    .to_string(),
            );
        }
        // SAFETY: a hint about this thread only; failing leaves it where the scheduler puts it
        unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
        let mut kperf = Self {
            kpc,
            kpep,
            db: ptr::null_mut(),
            config: ptr::null_mut(),
            slots: [None; COUNTERS],
            saved_force_all,
            efficiency_cpus: efficiency_cpus(),
        };
        kperf.configure()?;
        Ok(kperf)
    }

    /// Opens this CPU's database, programs [`EVENTS`] into the counters and starts them; `drop` undoes each step
    fn configure(&mut self) -> Result<(), String> {
        let (kpc, kpep) = (&self.kpc, &self.kpep);
        // SAFETY: each call gets live pointers from the steps before it and buffers of the size it is told
        unsafe {
            check(
                (kpep.db_create)(ptr::null(), &raw mut self.db),
                "opening the PMC database",
            )?;
            check(
                (kpep.config_create)(self.db, &raw mut self.config),
                "creating a config",
            )?;
            check(
                (kpep.config_force_counters)(self.config),
                "forcing the counters",
            )?;
            // indices of the events added, in the order the counter map lists them
            let mut added = Vec::with_capacity(COUNTERS);
            for (i, name) in EVENTS.iter().enumerate() {
                let mut event = ptr::null_mut();
                let found = (kpep.db_event)(self.db, name.as_ptr(), &raw mut event);
                if found != 0 && i >= REQUIRED {
                    continue;
                }
                check(found, &format!("finding {name:?} in the PMC database"))?;
                let status =
                    (kpep.config_add_event)(self.config, &raw mut event, 0, ptr::null_mut());
                if status != 0 && i >= REQUIRED {
                    continue;
                }
                check(status, &format!("adding {name:?}"))?;
                added.push(i);
            }
            let mut classes = 0;
            let mut registers = [0u64; MAX_COUNTERS];
            check(
                (kpep.config_kpc_classes)(self.config, &raw mut classes),
                "reading the counter classes",
            )?;
            let mut map = vec![0usize; added.len()];
            check(
                (kpep.config_kpc_map)(self.config, map.as_mut_ptr(), size_of_val(map.as_slice())),
                "mapping events to counters",
            )?;
            for (&event, &slot) in added.iter().zip(&map) {
                self.slots[event] = Some(slot);
            }
            let registers_size = size_of_val(&registers);
            check(
                (kpep.config_kpc)(self.config, registers.as_mut_ptr(), registers_size),
                "building the counter registers",
            )?;
            check(
                (kpc.force_all_ctrs_set)(1),
                "taking the counters from the system",
            )?;
            if classes & CLASS_CONFIGURABLE_MASK != 0 {
                check(
                    (kpc.set_config)(classes, registers.as_mut_ptr()),
                    "programming the counters",
                )?;
            }
            check((kpc.set_counting)(classes), "starting the counters")?;
            check(
                (kpc.set_thread_counting)(classes),
                "starting this thread's counters",
            )?;
        }
        Ok(())
    }

    /// Whether the calling thread runs on an efficiency core now
    pub fn on_efficiency_core(&self) -> bool {
        let mut cpu = 0;
        // SAFETY: writes the calling thread's CPU number into a local
        let status = unsafe { pthread_cpu_number_np(&raw mut cpu) };
        status == 0 && cpu < self.efficiency_cpus
    }

    /// The thread's running count of each event
    pub fn read(&self) -> Counts {
        let mut counters = [0u64; MAX_COUNTERS];
        // SAFETY: the buffer holds the MAX_COUNTERS values asked for; tid 0 is the calling thread
        let status = unsafe {
            (self.kpc.get_thread_counters)(0, MAX_COUNTERS as u32, counters.as_mut_ptr())
        };
        assert_eq!(
            status, 0,
            "kperf failed reading the counters, status {status}"
        );
        self.slots.map(|slot| slot.map_or(0, |slot| counters[slot]))
    }

    /// Which events the CPU counts, the others read as zero
    pub fn available(&self) -> [bool; COUNTERS] {
        self.slots.map(|slot| slot.is_some())
    }
}

impl Drop for Kperf {
    fn drop(&mut self) {
        // SAFETY: stops counting, restores the system's setting and frees what `configure` allocated, each once
        unsafe {
            (self.kpc.set_thread_counting)(0);
            (self.kpc.set_counting)(0);
            (self.kpc.force_all_ctrs_set)(self.saved_force_all);
            if !self.config.is_null() {
                (self.kpep.config_free)(self.config);
            }
            if !self.db.is_null() {
                (self.kpep.db_free)(self.db);
            }
        }
    }
}

/// Efficiency cores of the CPU, none on a CPU of a single core type
fn efficiency_cpus() -> usize {
    let read = |name: &CStr| {
        let mut value: c_int = 0;
        let mut size = size_of::<c_int>();
        // SAFETY: reads an integer sysctl into a local of its size
        let status = unsafe {
            sysctlbyname(
                name.as_ptr(),
                (&raw mut value).cast(),
                &raw mut size,
                ptr::null_mut(),
                0,
            )
        };
        if status == 0 { value as usize } else { 0 }
    };
    if read(c"hw.nperflevels") < 2 {
        return 0;
    }
    read(c"hw.perflevel1.logicalcpu")
}

/// A framework opened for good: its functions are called until the process exits
fn open(path: &CStr) -> Result<*mut c_void, String> {
    // SAFETY: a NUL-terminated path; the system frameworks run no unexpected initializers
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        return Err(format!("loading {path:?} failed"));
    }
    Ok(handle)
}

/// The function `name` of `handle`, as the function pointer type `F`
///
/// # Safety
///
/// `F` must be the `unsafe extern "C" fn` type of the symbol's real signature.
unsafe fn symbol<F: Copy>(handle: *mut c_void, name: &CStr) -> Result<F, String> {
    // SAFETY: the handle is a live `dlopen` handle and the name NUL-terminated
    let address = unsafe { dlsym(handle, name.as_ptr()) };
    if address.is_null() {
        return Err(format!("kperf has no function {name:?}"));
    }
    assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
    // SAFETY: a function pointer has the size of the data pointer `dlsym` returns, its type the caller's promise
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&address) })
}

fn check(status: c_int, step: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("kperf failed {step}, status {status}"))
    }
}
