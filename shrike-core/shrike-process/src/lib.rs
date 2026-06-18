//! Generic managed-subprocess lifecycle (#710, epic #703) — spawn, health-wait,
//! orphan reaping, escalating stop, extracted from `shrike-llama-server`'s
//! process half so any subprocess-managed runtime (a sync-server, a future
//! local model server) implements one small policy trait and inherits the
//! whole lifecycle.
//!
//! [`ManagedProcess`] is the policy seam: a runtime supplies binary resolution,
//! the argv vocabulary, the host/port it listens on, where it logs, and a
//! `health_check` hook — and [`Supervisor`] drives the rest. Keeping the health
//! probe a hook (not an HTTP call here) is what lets this crate stay at the
//! layer floor with **no HTTP dependency**: its deps are `shrike-error` + `libc`
//! + `tracing` only.
//!
//! The **orphan reaper is safety-critical** (#594/#654; see [`reaper`]): a
//! recorded PID is terminated only when it is BOTH still alive AND still the
//! process LISTENing on our port — both signals required, so a recycled PID can
//! never take a bystander down with it. The kill gate and the port→PID
//! attribution it rests on live in [`reaper`]; the [`Supervisor`] only invokes
//! the gate.
//!
//! `PR_SET_PDEATHSIG` is **intentionally avoided**: the parent-death signal keys
//! on the spawning *thread*, and a host typically starts a supervised process
//! under a pool thread (`asyncio.to_thread`), so a reclaimed pool thread could
//! kill a live server. The PID-file reaper is the deliberate alternative.

#![deny(missing_docs)]
#![deny(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use shrike_error::{ErrorKind, NativeError, NativeResult, ResultExt};

pub mod reaper;

use reaper::{pid_alive, port_bindable, terminate_raw, wait_pid_dead};
pub use reaper::{pid_owns_port, port_owner_pids};

/// How long [`Supervisor::start`] waits for the process to become healthy
/// before giving up and stopping it.
pub const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);
/// 50ms, not 250 (#426): a localhost health GET costs ~1ms, and every service
/// start rounds up to one poll quantum — a small model loads faster than a
/// single 250ms tick, so the coarser interval was pure added boot latency
/// (felt acutely by the embedding test suites, which boot servers per fixture).
pub const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// The SIGTERM grace window before escalating to SIGKILL.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
/// After a SIGKILL escalation death is fast — a killed process can't linger like
/// one ignoring SIGTERM. Also bounds the post-kill wait for the kernel to
/// release the orphan's listener.
pub const SIGKILL_TIMEOUT: Duration = Duration::from_secs(2);

/// The policy a managed subprocess supplies; [`Supervisor`] drives everything
/// else (spawn → reap → health-wait → stop, the PID file, the orphan reaper,
/// the non-blocking observer cell, best-effort `Drop`).
///
/// The seam is deliberately small: only the runtime-specific decisions live
/// here — *which* binary, *what* argv, *where* it listens and logs, and *how* to
/// probe health. The health probe is a hook (rather than an HTTP call in this
/// crate) so `shrike-process` carries no HTTP dependency; an HTTP-health policy
/// brings its own client.
pub trait ManagedProcess {
    /// Resolve the executable to spawn (override > env > PATH, validated). An
    /// unavailable binary is the policy's to report — typically
    /// [`NativeError::unavailable`].
    ///
    /// # Errors
    ///
    /// Returns an error (typically [`NativeError::unavailable`]) when the
    /// implementation cannot resolve a usable executable.
    fn binary(&self) -> NativeResult<String>;

    /// The exact argv (including `argv[0]`, the binary) to spawn. Shrike-owned
    /// flags first, any user passthrough last and reserved-flag-stripped — the
    /// policy owns that security guard.
    fn argv(&self, binary: &str) -> Vec<String>;

    /// The host the process listens on (for the port-bindable / reap probes and
    /// the base URL). A managed local server is loopback-pinned by policy.
    fn host(&self) -> &str;

    /// The port the process listens on — the reaper's identity key (a recorded
    /// PID is reaped only if it owns THIS port) and the bind-wait target.
    fn port(&self) -> u16;

    /// Where to record the child PID so a later start can reap an orphan left by
    /// an unclean shutdown (it survives a parent SIGKILL). `None` disables the
    /// reaper for this process.
    fn pid_file(&self) -> Option<&Path>;

    /// The directory for the process's own logs, if any. The supervisor creates
    /// it and appends the child's stderr to [`Self::stderr_log_name`] within it.
    fn log_dir(&self) -> Option<&Path> {
        None
    }

    /// The stderr log filename within [`Self::log_dir`].
    fn stderr_log_name(&self) -> &str {
        "process-stderr.log"
    }

    /// Probe liveness/readiness once. `base_url` is `http://{host}:{port}`. The
    /// supervisor polls this every [`HEALTH_POLL_INTERVAL`] until it returns
    /// `true`, the child exits, or [`HEALTH_TIMEOUT`] lapses. An HTTP policy GETs
    /// its health path here; another might probe a socket or a file.
    fn health_check(&self, base_url: &str) -> bool;

    /// A short noun for log lines (`"llama-server"`). Defaults generic.
    fn process_name(&self) -> &str {
        "managed process"
    }

    /// A one-line human description of what is being started (model, host, port,
    /// …) for the "Starting …" info log. Defaults to [`Self::process_name`].
    fn describe(&self) -> String {
        self.process_name().to_string()
    }
}

/// Drives a [`ManagedProcess`] through its lifecycle: spawn (reaping any orphan
/// first) → health-wait → stop (SIGTERM→SIGKILL). Generic over the policy so a
/// new managed runtime is a ~30-line `impl ManagedProcess` plus this supervisor.
///
/// `running`/`pid` are non-blocking even while a start holds the lifecycle: the
/// observed PID is mirrored in a [`pid_cell`](Self::pid_cell) a host can read
/// without contending with the (up to [`HEALTH_TIMEOUT`]-long) start.
pub struct Supervisor<P: ManagedProcess> {
    policy: P,
    child: Option<Child>,
    base_url: String,
    /// The observed child PID, shared with non-blocking observers: a host status
    /// path must NEVER contend with the lifecycle lock a 30s health-wait holds.
    /// Set at spawn, cleared on stop/observed exit; the micro-mutex is only ever
    /// held for a copy.
    pid_cell: Arc<Mutex<Option<u32>>>,
}

impl<P: ManagedProcess> Supervisor<P> {
    /// Wrap a policy. Nothing spawns until [`start`](Self::start).
    pub fn new(policy: P) -> Self {
        let base_url = format!("http://{}:{}", policy.host(), policy.port());
        Self {
            policy,
            child: None,
            base_url,
            pid_cell: Arc::default(),
        }
    }

    /// Borrow the policy (config inspection — e.g. a fingerprint built from the
    /// passthrough).
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Mutably borrow the policy to reshape it *before* spawning (e.g. switch an
    /// embeddings server to chat mode, add projectors). The cached `base_url` is
    /// derived from the host/port only, so an in-place reshape that doesn't move
    /// those keeps it valid; callers that would change host/port should rebuild
    /// the supervisor instead.
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// `http://{host}:{port}` — the endpoint the process serves.
    pub fn url(&self) -> &str {
        &self.base_url
    }

    /// The shared observed-PID cell: `Some` while a child is believed alive.
    /// Hosts read this instead of locking the supervisor when the lifecycle lock
    /// may be held.
    pub fn pid_cell(&self) -> Arc<Mutex<Option<u32>>> {
        Arc::clone(&self.pid_cell)
    }

    /// The live child's PID, or `None` when none is spawned.
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    /// True while the spawned child is alive (a poll, not a cached flag).
    ///
    /// # Panics
    ///
    /// Panics if the shared PID-cell mutex is poisoned (a prior holder panicked).
    pub fn running(&mut self) -> bool {
        let alive = match self.child.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        };
        if !alive {
            *self.pid_cell.lock().expect("pid cell poisoned") = None;
        }
        alive
    }

    /// Spawn the process and wait for it to become healthy. Reaps any orphan
    /// first. On health timeout the child is stopped and the error carries its
    /// exit code (if it died).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Unavailable`] if the policy cannot resolve its
    /// binary, the spawn fails, or the child does not become healthy within the
    /// health-wait window (the error carries the observed exit code).
    ///
    /// # Panics
    ///
    /// Panics if the shared PID-cell mutex is poisoned (a prior holder panicked).
    pub fn start(&mut self) -> NativeResult<()> {
        if self.running() {
            tracing::warn!(
                "{} already running (PID {:?})",
                self.policy.process_name(),
                self.pid()
            );
            return Ok(());
        }
        let binary = self.policy.binary()?;
        let argv = self.policy.argv(&binary);

        if let Some(dir) = self.policy.log_dir() {
            let _ = std::fs::create_dir_all(dir);
        }
        self.reap_orphan();

        tracing::info!(
            "Starting {}: {}",
            self.policy.process_name(),
            self.policy.describe()
        );

        let stderr: Stdio = match self.policy.log_dir() {
            Some(dir) => OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join(self.policy.stderr_log_name()))
                .map(Stdio::from)
                .unwrap_or_else(|_| Stdio::null()),
            None => Stdio::null(),
        };
        let child = Command::new(&argv[0])
            .args(&argv[1..])
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()
            .context(
                ErrorKind::Unavailable,
                format!("could not spawn {}", self.policy.process_name()),
            )?;
        *self.pid_cell.lock().expect("pid cell poisoned") = Some(child.id());
        self.child = Some(child);
        self.write_pid_file();

        if !self.wait_healthy() {
            // Read the exit code once, AFTER the stop — a pre-stop `try_wait`
            // reports a stale `None` for a child that exited microseconds later.
            let rc = self
                .stop_observing_exit()
                .map(|s| s.code().map_or("signal".to_string(), |c| c.to_string()))
                .unwrap_or_else(|| "None".to_string());
            return Err(NativeError::unavailable(format!(
                "{} failed to become healthy within {}s (exit code: {rc})",
                self.policy.process_name(),
                HEALTH_TIMEOUT.as_secs()
            )));
        }
        Ok(())
    }

    /// Poll the policy's `health_check` until it returns true, the child exits,
    /// or the timeout lapses.
    fn wait_healthy(&mut self) -> bool {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        while Instant::now() < deadline {
            if !self.running() {
                return false;
            }
            if self.policy.health_check(&self.base_url) {
                return true;
            }
            std::thread::sleep(HEALTH_POLL_INTERVAL);
        }
        false
    }

    /// Stop the child: SIGTERM, wait up to [`SHUTDOWN_TIMEOUT`], then SIGKILL.
    /// Clears the PID file.
    pub fn stop(&mut self) {
        let _ = self.stop_observing_exit();
    }

    /// [`stop`](Self::stop), reporting the child's exit status where it was
    /// observed — a genuine exit code for a child that had already died, a
    /// signal status for one our SIGTERM/SIGKILL took down.
    fn stop_observing_exit(&mut self) -> Option<ExitStatus> {
        *self.pid_cell.lock().expect("pid cell poisoned") = None;
        let mut child = self.child.take()?;
        let pid = child.id() as i64;
        let name = self.policy.process_name();
        if let Ok(Some(status)) = child.try_wait() {
            tracing::info!("{name} already exited (PID {pid}, code {status:?})");
            self.clear_pid_file();
            return Some(status);
        }
        tracing::info!("Stopping {name} (PID {pid})");
        terminate_raw(pid, false);
        if !wait_child(&mut child, SHUTDOWN_TIMEOUT) {
            tracing::warn!("{name} did not exit after SIGTERM, sending SIGKILL");
            let _ = child.kill();
            wait_child(&mut child, SIGKILL_TIMEOUT);
        }
        tracing::info!("{name} stopped (PID {pid})");
        self.clear_pid_file();
        child.try_wait().ok().flatten()
    }

    fn write_pid_file(&self) {
        let (Some(path), Some(child)) = (self.policy.pid_file(), &self.child) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::File::create(path).and_then(|mut f| write!(f, "{}", child.id()));
    }

    fn clear_pid_file(&self) {
        if let Some(path) = self.policy.pid_file() {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Kill a process left over from a prior unclean shutdown. The orphan is the
    /// recorded PID **only if that PID is itself the process LISTENing on our
    /// port** — checked with [`pid_owns_port`], not an independent `pid_alive &&
    /// port_held` pair. That pair could kill an unrelated process: a recycled PID
    /// (now some other live process) plus *any* unrelated holder of our port
    /// satisfied both signals without the PID ever being the holder. Requiring
    /// the PID to own the port closes that, and fails safe (no positive ownership
    /// → no kill) when ownership can't be established. Private: it clears the PID
    /// file as a side effect, so it is strictly a pre-spawn step of
    /// [`start`](Self::start) — calling it with a live child would wipe that
    /// child's reap record.
    fn reap_orphan(&self) {
        let Some(path) = self.policy.pid_file() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let Ok(pid) = text.trim().parse::<i64>() else {
            self.clear_pid_file();
            return;
        };
        // `pid_owns_port` already implies the PID is alive (it holds a socket);
        // the cheap `pid_alive` short-circuits the lsof spawn for an
        // obviously-dead recorded PID.
        if pid_alive(pid) && pid_owns_port(pid, self.policy.port()) {
            tracing::warn!(
                "Reaping orphaned {} (PID {pid}) holding port {}",
                self.policy.process_name(),
                self.policy.port()
            );
            self.terminate_pid(pid);
        }
        self.clear_pid_file();
    }

    /// SIGTERM, then SIGKILL, a stale PID — confirming death via `pid_alive`
    /// going false (never the port: an unrelated process could grab the freed
    /// port mid-window and read as "kill failed"), then waiting for the port to
    /// become bindable for the spawn that follows.
    fn terminate_pid(&self, pid: i64) {
        let name = self.policy.process_name();
        terminate_raw(pid, false);
        if !wait_pid_dead(pid, SHUTDOWN_TIMEOUT) {
            tracing::warn!("Orphan {name} (PID {pid}) ignored SIGTERM, sending SIGKILL");
            terminate_raw(pid, true);
            if !wait_pid_dead(pid, SIGKILL_TIMEOUT) {
                tracing::warn!("Orphan {name} (PID {pid}) survived SIGKILL");
            }
        }
        // Death (dis)confirmed; separately wait for the kernel to release the
        // orphan's listener so the spawn that follows can bind.
        self.wait_port_bindable(SIGKILL_TIMEOUT);
    }

    /// Poll until the port binds. A just-released port can be transiently
    /// unbindable (TIME_WAIT / close→rebind races), so this can spin briefly even
    /// after the holder's death is already confirmed.
    fn wait_port_bindable(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if port_bindable(self.policy.host(), self.policy.port()) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        port_bindable(self.policy.host(), self.policy.port())
    }
}

impl<P: ManagedProcess> Drop for Supervisor<P> {
    fn drop(&mut self) {
        // The child is the host's direct responsibility; a dropped supervisor
        // must not leak the process (the PID file still allows reaping if the
        // whole process dies before Drop). NOTE: with a live child this can block
        // several seconds — the SIGTERM wait ([`SHUTDOWN_TIMEOUT`]) plus the
        // SIGKILL wait ([`SIGKILL_TIMEOUT`]).
        if self.child.is_some() {
            self.stop();
        }
    }
}

/// Poll-wait for child exit (std has no `wait_timeout`).
fn wait_child(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    matches!(child.try_wait(), Ok(Some(_)))
}

/// Whether `path` is an executable file (any execute bit on unix; a plain file
/// off unix). Reusable binary-resolution helper for a [`ManagedProcess::binary`]
/// impl.
pub fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// A minimal PATH search (avoids a deps-for-one-call crate). Returns the first
/// PATH entry holding an executable `name`. Reusable binary-resolution helper.
pub fn which(name: &str) -> Option<String> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() && is_executable(&candidate) {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A minimal in-test policy: no real binary, a sleeper argv, configurable
    /// health, used to exercise the generic lifecycle (spawn/stop/reap/pid_cell)
    /// independent of any concrete runtime.
    struct TestPolicy {
        host: String,
        port: u16,
        pid_file: Option<PathBuf>,
    }

    impl TestPolicy {
        fn new(port: u16) -> Self {
            Self {
                host: "127.0.0.1".into(),
                port,
                pid_file: None,
            }
        }
    }

    impl ManagedProcess for TestPolicy {
        fn binary(&self) -> NativeResult<String> {
            Ok("/bin/sleep".into())
        }
        fn argv(&self, binary: &str) -> Vec<String> {
            vec![binary.into(), "30".into()]
        }
        fn host(&self) -> &str {
            &self.host
        }
        fn port(&self) -> u16 {
            self.port
        }
        fn pid_file(&self) -> Option<&Path> {
            self.pid_file.as_deref()
        }
        fn health_check(&self, _base_url: &str) -> bool {
            true
        }
        fn process_name(&self) -> &str {
            "test-process"
        }
    }

    fn supervisor(port: u16) -> Supervisor<TestPolicy> {
        Supervisor::new(TestPolicy::new(port))
    }

    #[test]
    fn url_is_host_and_port() {
        let s = supervisor(18373);
        assert_eq!(s.url(), "http://127.0.0.1:18373");
    }

    #[test]
    fn stale_pid_file_is_cleared_without_killing() {
        let dir = std::env::temp_dir().join(format!("shrike-reap-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("process.pid");
        // Garbage content → cleared.
        std::fs::write(&pid_file, "not-a-pid").unwrap();
        let mut s = supervisor(18373);
        s.policy.pid_file = Some(pid_file.clone());
        s.reap_orphan();
        assert!(!pid_file.exists());
        // A dead PID (no such process) → cleared, no port wait.
        std::fs::write(&pid_file, "999999").unwrap();
        s.reap_orphan();
        assert!(!pid_file.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn alive_pid_without_the_port_is_not_reaped() {
        // The dual signal: a live process whose PID is recorded but that does
        // NOT hold our port (recycled PID case) must survive.
        let dir = std::env::temp_dir().join(format!("shrike-reap2-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("process.pid");
        let mut sleeper = Command::new("/bin/sleep").arg("30").spawn().unwrap();
        std::fs::write(&pid_file, sleeper.id().to_string()).unwrap();
        let mut s = supervisor(18373); // port 18373 — nothing listens there
        s.policy.pid_file = Some(pid_file.clone());
        s.reap_orphan();
        assert!(!pid_file.exists(), "the stale record is always cleared");
        assert!(
            matches!(sleeper.try_wait(), Ok(None)),
            "the live non-holder was not killed"
        );
        let _ = sleeper.kill();
        let _ = sleeper.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn s12_recycled_pid_with_unrelated_port_holder_is_not_killed() {
        // #594 (audit S12-1): a recycled PID recorded for reaping, plus an
        // UNRELATED process holding our port — the old `pid_alive(recorded) &&
        // port_held(ANY holder)` pair killed the bystander because it never
        // asked whether the recorded PID *is* the holder. The unrelated process
        // (U) must survive: it does not own our port → not an orphan. This pins
        // the both-signals-required invariant at the Supervisor's reap gate.
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port(); // H holds this port

        let dir = std::env::temp_dir().join(format!("shrike-s12-wrongkill-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("process.pid");
        // U: an unrelated live process whose PID landed in our reap record.
        let mut victim = Command::new("/bin/sleep").arg("30").spawn().unwrap();
        std::fs::write(&pid_file, victim.id().to_string()).unwrap();

        let mut s = supervisor(port); // reap against the port H holds
        s.policy.pid_file = Some(pid_file.clone());

        s.reap_orphan();

        // U must survive (it does not hold our port → not an orphan).
        let victim_status = victim.try_wait();
        let _ = victim.kill();
        let _ = victim.wait();
        drop(listener);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            matches!(victim_status, Ok(None)),
            "the recycled-PID unrelated process must survive (it does not hold our port); \
             got {victim_status:?} — the dual signal killed the wrong process"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_real_orphan_holding_our_port_is_still_reaped() {
        // The boundary: the fix must not regress the legitimate reap. A
        // *detached* (reparented-to-init) process that genuinely LISTENs on our
        // port — a real orphan — and whose PID is recorded must still be
        // terminated. Detach via `sh -c '... & echo $!'` so the SIGTERM'd
        // process is reaped by init rather than zombied (a direct child would
        // linger as a zombie that `kill(pid, 0)` still reports alive).
        // Capability gate (#652): this verifies a *positive* reap, which needs to
        // confirm ownership via `port_owner_pids` (lsof on unix). The bazel
        // darwin-sandbox provides no `lsof`, so ownership can never be
        // established there and the setup precondition could never hold. Probe
        // the capability *positively* — bind an in-process listener and ask
        // whether we can observe our OWN PID owning it. If we cannot, the
        // platform can't introspect port ownership here, so skip cleanly (a
        // returning Rust test is reported PASSED). Keying the skip on this
        // positive incapability — not on catching the would-be setup failure —
        // keeps a genuine reap regression FAILING rather than silently skipped:
        // wherever lsof exists (dev machines, CI Linux) the probe passes and the
        // full reap is exercised. The production reap is unaffected — it already
        // treats "ownership unprovable" as "never kill" (fail-open, gated).
        {
            let self_listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let self_port = self_listener.local_addr().unwrap().port();
            let self_pid = std::process::id() as i64;
            let observable = pid_owns_port(self_pid, self_port);
            drop(self_listener);
            if !observable {
                eprintln!(
                    "SKIP a_real_orphan_holding_our_port_is_still_reaped: port ownership is \
                     not observable here (no lsof/netstat in this sandbox) — the positive-reap \
                     assertion cannot be verified. See #652."
                );
                return;
            }
        }

        let dir = std::env::temp_dir().join(format!("shrike-s12-orphan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("process.pid");
        // Pick a free port, then hand it to the detached listener.
        let probe = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        let py = format!(
            "import socket,time;\
             s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);\
             s.bind(('127.0.0.1',{port}));s.listen();time.sleep(30)"
        );
        let out = Command::new("/bin/sh")
            .args([
                "-c",
                &format!("python3 -c \"{py}\" >/dev/null 2>&1 & echo $!"),
            ])
            .output()
            .unwrap();
        let pid: i64 = String::from_utf8(out.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        std::fs::write(&pid_file, pid.to_string()).unwrap();

        // Wait until the orphan is actually LISTENing (it owns the port).
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !pid_owns_port(pid, port) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(
            pid_owns_port(pid, port),
            "test setup: orphan never bound the port"
        );

        let mut s = supervisor(port);
        s.policy.pid_file = Some(pid_file.clone());
        s.reap_orphan();

        // The genuine orphan was terminated and the record cleared.
        let dead = !pid_alive(pid);
        if !dead {
            terminate_raw(pid, true); // belt-and-suspenders cleanup
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(dead, "a real orphan holding our port must still be reaped");
        assert!(!pid_file.exists(), "the record is cleared after the reap");
    }

    #[cfg(unix)]
    #[test]
    fn terminate_pid_confirms_death_via_pid_not_port() {
        // A detached (reparented-to-init) sleeper, like a real orphan — our own
        // child would zombie until waited, and `kill(pid, 0)` would still see it.
        let out = Command::new("/bin/sh")
            .args(["-c", "sleep 30 >/dev/null 2>&1 & echo $!"])
            .output()
            .unwrap();
        let pid: i64 = String::from_utf8(out.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert!(pid_alive(pid));
        let s = supervisor(18373); // port 18373 — free throughout
        let started = Instant::now();
        s.terminate_pid(pid);
        assert!(!pid_alive(pid), "death confirmed via the PID");
        // SIGTERM landed and was confirmed promptly — no SIGKILL tier, and no
        // connect-timeout padding from a port-based confirmation.
        assert!(started.elapsed() < SHUTDOWN_TIMEOUT);
    }

    #[cfg(unix)]
    #[test]
    fn pid_cell_tracks_spawn_exit_and_stop() {
        let mut s = supervisor(18373);
        let cell = s.pid_cell();
        assert_eq!(*cell.lock().unwrap(), None);
        let child = Command::new("/bin/sleep").arg("30").spawn().unwrap();
        *cell.lock().unwrap() = Some(child.id());
        s.child = Some(child);
        assert!(s.running());
        assert!(cell.lock().unwrap().is_some());
        s.stop();
        assert_eq!(*cell.lock().unwrap(), None, "stop clears the cell");
        // An observed exit clears it too.
        let mut quick = Command::new("/bin/sleep").arg("0").spawn().unwrap();
        let _ = quick.wait();
        s.child = Some(quick);
        *cell.lock().unwrap() = Some(1);
        assert!(!s.running());
        assert_eq!(*cell.lock().unwrap(), None, "observed exit clears the cell");
    }

    #[cfg(unix)]
    #[test]
    fn stop_terminates_a_live_child_and_clears_the_pid_file() {
        let dir = std::env::temp_dir().join(format!("shrike-stop-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("process.pid");
        let mut s = supervisor(18373);
        s.policy.pid_file = Some(pid_file.clone());
        s.child = Some(Command::new("/bin/sleep").arg("30").spawn().unwrap());
        s.write_pid_file();
        assert!(s.running());
        assert!(pid_file.exists());
        s.stop();
        assert!(!s.running());
        assert!(!pid_file.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn stop_escalates_to_sigkill_for_a_term_ignoring_child() {
        // A different, dedicated policy whose argv installs a TERM trap, so the
        // generic stop escalation is exercised end to end.
        struct TrapPolicy;
        impl ManagedProcess for TrapPolicy {
            fn binary(&self) -> NativeResult<String> {
                Ok("/bin/sh".into())
            }
            fn argv(&self, binary: &str) -> Vec<String> {
                vec![binary.into(), "-c".into(), "trap '' TERM; sleep 30".into()]
            }
            fn host(&self) -> &str {
                "127.0.0.1"
            }
            fn port(&self) -> u16 {
                18373
            }
            fn pid_file(&self) -> Option<&Path> {
                None
            }
            fn health_check(&self, _base_url: &str) -> bool {
                true
            }
        }
        let mut s = Supervisor::new(TrapPolicy);
        s.child = Some(
            Command::new("/bin/sh")
                .args(["-c", "trap '' TERM; sleep 30"])
                .spawn()
                .unwrap(),
        );
        // Give the shell a beat to install the trap.
        std::thread::sleep(Duration::from_millis(200));
        let started = Instant::now();
        s.stop();
        assert!(!s.running());
        // It waited out SHUTDOWN_TIMEOUT before the kill tier.
        assert!(started.elapsed() >= SHUTDOWN_TIMEOUT);
    }
}
