//! llama-server subprocess lifecycle (#342 P4b) — the 1:1 port of
//! `LlamaServerBackend`'s process half. Spawn with the exact command
//! construction and the reserved-flag security guardrail, wait for health,
//! reap orphans from prior unclean shutdowns, stop with SIGTERM→SIGKILL
//! escalation. The manager is NOT an embedder: it produces a healthy
//! loopback endpoint that the host hands to `shrike-embed-remote`.
//!
//! Security boundary preserved verbatim: `--host` is Shrike-owned — the
//! embedding backend is deliberately pinned to loopback (audit §1.1) — so
//! the generic passthrough strips every reserved flag (with its value token
//! for the value-taking ones) before the command is built.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use shrike_ffi::{NativeError, NativeResult};

pub const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);
pub const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
/// After a SIGKILL escalation death is fast — a killed process can't linger
/// like one ignoring SIGTERM. Also bounds the post-kill wait for the kernel
/// to release the orphan's listener.
pub const SIGKILL_TIMEOUT: Duration = Duration::from_secs(2);

/// llama-server flags Shrike owns; the generic passthrough must not override
/// them. `--embedding` is llama.cpp's alias for `--embeddings`.
pub const RESERVED_FLAGS: &[&str] = &[
    "--model",
    "-m",
    "--host",
    "--port",
    "--embeddings",
    "--embedding",
];
/// Of the reserved flags, those that consume a following value token (so a
/// rejected `--host 0.0.0.0` drops the value too, not just the flag).
pub const RESERVED_VALUE_FLAGS: &[&str] = &["--model", "-m", "--host", "--port"];

pub struct LlamaServerConfig {
    /// Explicit binary override (`--llama-server`); beats env and PATH.
    pub binary: Option<String>,
    pub model: String,
    pub host: String,
    pub port: u16,
    pub log_dir: Option<PathBuf>,
    pub context_size: Option<u32>,
    pub threads: Option<u32>,
    pub gpu_layers: Option<i32>,
    pub pooling: Option<String>,
    /// Raw passthrough entries (each shlex-split at command-build time).
    pub extra_args: Vec<String>,
    /// Records the child PID so a later start can reap an orphan left by an
    /// unclean Shrike shutdown (it survives a parent SIGKILL).
    pub pid_file: Option<PathBuf>,
}

pub struct LlamaServerManager {
    cfg: LlamaServerConfig,
    child: Option<Child>,
    base_url: String,
    /// The observed child PID, shared with non-blocking observers: a host
    /// status path must NEVER contend with the lifecycle lock a 30s
    /// health-wait holds (the Python facade's `running` was a lock-free
    /// `poll()`; this cell keeps that property). Set at spawn, cleared on
    /// stop/observed exit; the micro-mutex is only ever held for a copy.
    pid_cell: std::sync::Arc<std::sync::Mutex<Option<u32>>>,
}

fn pid_alive(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // kill(pid, 0): 0 = exists; EPERM = exists but not ours.
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        // Best effort off unix: no cheap existence probe — the port half of
        // the dual signal carries the reap decision alone.
        true
    }
}

fn terminate_raw(pid: i64, hard: bool) {
    #[cfg(unix)]
    {
        let sig = if hard { libc::SIGKILL } else { libc::SIGTERM };
        unsafe {
            libc::kill(pid as libc::pid_t, sig);
        }
    }
    #[cfg(not(unix))]
    {
        // Windows has no graceful tier from outside the handle (Python's
        // os.kill maps to TerminateProcess the same way).
        let _ = hard;
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// "Is the port available to bind right now" — a `TcpListener::bind` probe
/// (instant; a connect probe's 500ms timeout would dominate any wait loop
/// built on it). The probe listener is dropped immediately.
fn port_bindable(host: &str, port: u16) -> bool {
    TcpListener::bind((host, port)).is_ok()
}

/// Something else holds the port — `EADDRINUSE` specifically, so an
/// unrelated bind failure (e.g. an unresolvable host) never reads as "held"
/// and corroborates a kill.
fn port_held(host: &str, port: u16) -> bool {
    matches!(
        TcpListener::bind((host, port)),
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse
    )
}

/// Poll-wait for a PID to die. This is the kill confirmation: `pid_alive`
/// going false, never the port — port state is not process identity (an
/// unrelated process can grab the freed port mid-window). Only meaningful
/// for a *non-child* PID (a prior process's orphan, reparented to init and
/// reaped there); our own child would zombie until waited.
fn wait_pid_dead(pid: i64, timeout: Duration) -> bool {
    #[cfg(unix)]
    {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if !pid_alive(pid) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        !pid_alive(pid)
    }
    #[cfg(not(unix))]
    {
        // taskkill /F already terminated forcefully, and `pid_alive` has no
        // cheap existence probe off unix — nothing to poll.
        let _ = (pid, timeout);
        true
    }
}

impl LlamaServerManager {
    pub fn new(cfg: LlamaServerConfig) -> Self {
        let base_url = format!("http://{}:{}", cfg.host, cfg.port);
        Self {
            cfg,
            child: None,
            base_url,
            pid_cell: std::sync::Arc::default(),
        }
    }

    /// The shared observed-PID cell (see the field docs): `Some` while a
    /// child is believed alive. Hosts read this instead of locking the
    /// manager when the lifecycle lock may be held.
    pub fn pid_cell(&self) -> std::sync::Arc<std::sync::Mutex<Option<u32>>> {
        std::sync::Arc::clone(&self.pid_cell)
    }

    pub fn url(&self) -> &str {
        &self.base_url
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    /// True while the spawned child is alive (a poll, not a cached flag).
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

    /// Locate the llama-server binary: explicit override (validated) >
    /// `LLAMA_SERVER_PATH` (validated) > `PATH`.
    pub fn find_binary(&self) -> NativeResult<String> {
        let env_path = self.cfg.binary.clone().or_else(|| {
            std::env::var("LLAMA_SERVER_PATH")
                .ok()
                .filter(|s| !s.is_empty())
        });
        if let Some(p) = env_path {
            let path = Path::new(&p);
            if path.is_file() && is_executable(path) {
                return Ok(p);
            }
            return Err(NativeError::unavailable(format!(
                "LLAMA_SERVER_PATH={p} does not point to an executable"
            )));
        }
        if let Some(found) = which("llama-server") {
            return Ok(found);
        }
        Err(NativeError::unavailable(
            "llama-server not found. Install llama.cpp or set LLAMA_SERVER_PATH.",
        ))
    }

    /// Resolve `extra_args` to llama-server tokens, dropping reserved flags
    /// — including a separate value token for value-taking flags (`--host
    /// 0.0.0.0` loses both; the self-contained `--host=0.0.0.0` is one
    /// token). `warn` logs each rejection (the command-build path); the
    /// fingerprint reuses this silently.
    pub fn passthrough_tokens(&self, warn: bool) -> Vec<String> {
        let mut raw: Vec<String> = Vec::new();
        for entry in &self.cfg.extra_args {
            if let Some(tokens) = shlex::split(entry) {
                raw.extend(tokens);
            }
        }
        let mut result = Vec::new();
        let mut i = 0;
        while i < raw.len() {
            let tok = &raw[i];
            let flag = tok.split('=').next().unwrap_or(tok);
            if RESERVED_FLAGS.contains(&flag) {
                if warn {
                    tracing::warn!(
                        "Ignoring reserved llama-server flag {flag:?} passed via \
                         --embedding-arg; Shrike controls it (use a typed setting for \
                         vector-affecting flags)."
                    );
                }
                if RESERVED_VALUE_FLAGS.contains(&flag) && !tok.contains('=') && i + 1 < raw.len() {
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }
            result.push(tok.clone());
            i += 1;
        }
        result
    }

    /// The exact command, Shrike-owned flags first, passthrough last (so a
    /// user can't shadow Shrike's args by ordering; reserved flags are
    /// stripped regardless).
    pub fn build_command(&self, binary: &str) -> Vec<String> {
        let mut cmd: Vec<String> = vec![
            binary.to_string(),
            "--model".into(),
            self.cfg.model.clone(),
            "--host".into(),
            self.cfg.host.clone(),
            "--port".into(),
            self.cfg.port.to_string(),
            "--embeddings".into(),
        ];
        if let Some(n) = self.cfg.context_size {
            cmd.extend(["--ctx-size".into(), n.to_string()]);
        }
        if let Some(n) = self.cfg.threads {
            cmd.extend(["--threads".into(), n.to_string()]);
        }
        if let Some(n) = self.cfg.gpu_layers {
            cmd.extend(["--gpu-layers".into(), n.to_string()]);
        }
        if let Some(p) = &self.cfg.pooling {
            // Override the GGUF's stored pooling type — required for
            // last-token models whose metadata omits it.
            cmd.extend(["--pooling".into(), p.clone()]);
        }
        if let Some(dir) = &self.cfg.log_dir {
            cmd.extend([
                "--log-file".into(),
                dir.join("llama-server.log").to_string_lossy().into_owned(),
            ]);
        }
        cmd.extend(self.passthrough_tokens(true));
        cmd
    }

    fn write_pid_file(&self) {
        let (Some(path), Some(child)) = (&self.cfg.pid_file, &self.child) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::File::create(path).and_then(|mut f| write!(f, "{}", child.id()));
    }

    fn clear_pid_file(&self) {
        if let Some(path) = &self.cfg.pid_file {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Kill a llama-server left over from a prior unclean shutdown. A
    /// recorded PID that is still alive *and* holding our port is an orphan;
    /// both signals are required so a recycled PID can't make us kill an
    /// unrelated process. Private: it clears the PID file as a side effect,
    /// so it is strictly a pre-spawn step of [`start`](Self::start) — a host
    /// calling it with a live child would wipe that child's reap record.
    fn reap_orphan(&self) {
        let Some(path) = &self.cfg.pid_file else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let Ok(pid) = text.trim().parse::<i64>() else {
            self.clear_pid_file();
            return;
        };
        if pid_alive(pid) && port_held(&self.cfg.host, self.cfg.port) {
            tracing::warn!(
                "Reaping orphaned llama-server (PID {pid}) holding port {}",
                self.cfg.port
            );
            self.terminate_pid(pid);
        }
        self.clear_pid_file();
    }

    /// SIGTERM, then SIGKILL, a stale PID — confirming death via
    /// `pid_alive` going false (never the port: an unrelated process could
    /// grab the freed port mid-window and read as "kill failed"), then
    /// waiting for the port to become bindable for the spawn that follows.
    fn terminate_pid(&self, pid: i64) {
        terminate_raw(pid, false);
        if !wait_pid_dead(pid, SHUTDOWN_TIMEOUT) {
            tracing::warn!("Orphan llama-server (PID {pid}) ignored SIGTERM, sending SIGKILL");
            terminate_raw(pid, true);
            if !wait_pid_dead(pid, SIGKILL_TIMEOUT) {
                tracing::warn!("Orphan llama-server (PID {pid}) survived SIGKILL");
            }
        }
        // Death (dis)confirmed; separately wait for the kernel to release
        // the orphan's listener so the spawn that follows can bind.
        self.wait_port_bindable(SIGKILL_TIMEOUT);
    }

    fn wait_port_bindable(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if port_bindable(&self.cfg.host, self.cfg.port) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        port_bindable(&self.cfg.host, self.cfg.port)
    }

    /// Spawn llama-server and wait for it to become healthy. Reaps any
    /// orphan first. On health timeout the child is stopped and the error
    /// carries its exit code (if it died).
    pub fn start(&mut self) -> NativeResult<()> {
        if self.running() {
            tracing::warn!("Embedding service already running (PID {:?})", self.pid());
            return Ok(());
        }
        let binary = self.find_binary()?;
        let argv = self.build_command(&binary);

        if let Some(dir) = &self.cfg.log_dir {
            let _ = std::fs::create_dir_all(dir);
        }
        self.reap_orphan();

        tracing::info!(
            "Starting llama-server: model={}, host={}, port={}",
            self.cfg.model,
            self.cfg.host,
            self.cfg.port
        );

        let stderr: Stdio = match &self.cfg.log_dir {
            Some(dir) => OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("llama-server-stderr.log"))
                .map(Stdio::from)
                .unwrap_or_else(|_| Stdio::null()),
            None => Stdio::null(),
        };
        let child = Command::new(&argv[0])
            .args(&argv[1..])
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()
            .map_err(|e| NativeError::unavailable(format!("could not spawn llama-server: {e}")))?;
        *self.pid_cell.lock().expect("pid cell poisoned") = Some(child.id());
        self.child = Some(child);
        self.write_pid_file();

        if !self.wait_healthy() {
            // Read the exit code once, AFTER the stop — a pre-stop
            // `try_wait` reports a stale `None` for a child that exited
            // microseconds later.
            let rc = self
                .stop_observing_exit()
                .map(|s| s.code().map_or("signal".to_string(), |c| c.to_string()))
                .unwrap_or_else(|| "None".to_string());
            return Err(NativeError::unavailable(format!(
                "llama-server failed to become healthy within {}s (exit code: {rc})",
                HEALTH_TIMEOUT.as_secs()
            )));
        }
        Ok(())
    }

    /// Poll `GET /health` until 200, the child exits, or the timeout lapses.
    fn wait_healthy(&mut self) -> bool {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(2))
            .build();
        let url = format!("{}/health", self.base_url);
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        while Instant::now() < deadline {
            if !self.running() {
                return false;
            }
            if matches!(agent.get(&url).call(), Ok(r) if r.status() == 200) {
                return true;
            }
            std::thread::sleep(HEALTH_POLL_INTERVAL);
        }
        false
    }

    /// Stop the child: SIGTERM, wait up to [`SHUTDOWN_TIMEOUT`], then
    /// SIGKILL. Clears the PID file.
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
        if let Ok(Some(status)) = child.try_wait() {
            tracing::info!("Embedding service already exited (PID {pid}, code {status:?})");
            self.clear_pid_file();
            return Some(status);
        }
        tracing::info!("Stopping embedding service (PID {pid})");
        terminate_raw(pid, false);
        if !wait_child(&mut child, SHUTDOWN_TIMEOUT) {
            tracing::warn!("llama-server did not exit after SIGTERM, sending SIGKILL");
            let _ = child.kill();
            wait_child(&mut child, SIGKILL_TIMEOUT);
        }
        tracing::info!("Embedding service stopped (PID {pid})");
        self.clear_pid_file();
        child.try_wait().ok().flatten()
    }
}

impl Drop for LlamaServerManager {
    fn drop(&mut self) {
        // The child is Shrike's direct responsibility; a dropped manager
        // must not leak a llama-server (the PID file still allows reaping
        // if the whole process dies before Drop). NOTE: with a live child
        // this can block several seconds — the SIGTERM wait
        // ([`SHUTDOWN_TIMEOUT`]) plus the SIGKILL wait ([`SIGKILL_TIMEOUT`]).
        if self.child.is_some() {
            self.stop();
        }
    }
}

/// Poll-wait for child exit (std has no wait_timeout).
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

fn is_executable(path: &Path) -> bool {
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

/// A minimal PATH search (avoids a deps-for-one-call crate).
fn which(name: &str) -> Option<String> {
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
    use std::sync::Mutex;

    /// Env mutation is process-global — serialize the tests that touch it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn manager(extra: &[&str]) -> LlamaServerManager {
        LlamaServerManager::new(LlamaServerConfig {
            binary: None,
            model: "/models/test.gguf".into(),
            host: "127.0.0.1".into(),
            port: 18373,
            log_dir: None,
            context_size: None,
            threads: None,
            gpu_layers: None,
            pooling: None,
            extra_args: extra.iter().map(|s| s.to_string()).collect(),
            pid_file: None,
        })
    }

    #[test]
    fn command_construction_is_exact_and_passthrough_last() {
        let mut m = manager(&["--flash-attn --ubatch-size 256"]);
        m.cfg.context_size = Some(512);
        m.cfg.pooling = Some("last".into());
        let cmd = m.build_command("/bin/llama-server");
        assert_eq!(
            cmd,
            vec![
                "/bin/llama-server",
                "--model",
                "/models/test.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "18373",
                "--embeddings",
                "--ctx-size",
                "512",
                "--pooling",
                "last",
                "--flash-attn",
                "--ubatch-size",
                "256",
            ]
        );
    }

    #[test]
    fn reserved_flags_are_stripped_with_their_values() {
        // `--host 0.0.0.0` loses flag AND value; `--port=9` is one token;
        // a trailing reserved flag with no value doesn't panic.
        let m = manager(&["--host 0.0.0.0 --flash-attn", "--port=9999", "-m"]);
        assert_eq!(m.passthrough_tokens(false), vec!["--flash-attn"]);
    }

    #[test]
    fn binary_resolution_precedence_and_validation() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("shrike-llama-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("llama-server");
        std::fs::write(&bin, "#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        // Explicit override wins and is validated.
        let mut m = manager(&[]);
        m.cfg.binary = Some(bin.to_string_lossy().into_owned());
        assert_eq!(m.find_binary().unwrap(), bin.to_string_lossy());

        // A bad env path errors loud, never falls through to PATH.
        std::env::set_var("LLAMA_SERVER_PATH", "/nonexistent/binary");
        let m = manager(&[]);
        let err = m.find_binary().expect_err("must fail");
        assert!(err.to_string().contains("does not point to an executable"));
        std::env::remove_var("LLAMA_SERVER_PATH");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_pid_file_is_cleared_without_killing() {
        let dir = std::env::temp_dir().join(format!("shrike-reap-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("embedding.pid");
        // Garbage content → cleared.
        std::fs::write(&pid_file, "not-a-pid").unwrap();
        let mut m = manager(&[]);
        m.cfg.pid_file = Some(pid_file.clone());
        m.reap_orphan();
        assert!(!pid_file.exists());
        // A dead PID (no such process) → cleared, no port wait.
        std::fs::write(&pid_file, "999999").unwrap();
        m.reap_orphan();
        assert!(!pid_file.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn alive_pid_without_the_port_is_not_reaped() {
        // The dual signal: a live process whose PID is recorded but that
        // does NOT hold our port (recycled PID case) must survive.
        let dir = std::env::temp_dir().join(format!("shrike-reap2-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("embedding.pid");
        let mut sleeper = Command::new("/bin/sleep").arg("30").spawn().unwrap();
        std::fs::write(&pid_file, sleeper.id().to_string()).unwrap();
        let mut m = manager(&[]); // port 18373 — nothing listens there
        m.cfg.pid_file = Some(pid_file.clone());
        m.reap_orphan();
        assert!(!pid_file.exists(), "the stale record is always cleared");
        assert!(
            matches!(sleeper.try_wait(), Ok(None)),
            "the live non-holder was not killed"
        );
        let _ = sleeper.kill();
        let _ = sleeper.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bind_probe_distinguishes_free_held_and_unresolvable() {
        // A fixed port below the ephemeral range (an ephemeral one, once
        // dropped, can be re-handed to any concurrent outbound connection
        // system-wide), distinct from the 18373 the other tests probe.
        let port = 18374;
        // Free: bindable, not held.
        assert!(port_bindable("127.0.0.1", port));
        assert!(!port_held("127.0.0.1", port));
        // Held: a live listener → not bindable, held (EADDRINUSE).
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
        assert!(!port_bindable("127.0.0.1", port));
        assert!(port_held("127.0.0.1", port));
        // Released: bindable again, no longer held.
        drop(listener);
        assert!(port_bindable("127.0.0.1", port));
        assert!(!port_held("127.0.0.1", port));
        // An unresolvable host is NOT "held" — a bind failure that isn't
        // EADDRINUSE must never corroborate a kill.
        assert!(!port_held("host.invalid.shrike-test", port));
    }

    #[cfg(unix)]
    #[test]
    fn terminate_pid_confirms_death_via_pid_not_port() {
        // A detached (reparented-to-init) sleeper, like a real orphan — our
        // own child would zombie until waited, and `kill(pid, 0)` would
        // still see it.
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
        let m = manager(&[]); // port 18373 — free throughout
        let started = Instant::now();
        m.terminate_pid(pid);
        assert!(!pid_alive(pid), "death confirmed via the PID");
        // SIGTERM landed and was confirmed promptly — no SIGKILL tier, and
        // no connect-timeout padding from a port-based confirmation.
        assert!(started.elapsed() < SHUTDOWN_TIMEOUT);
    }

    #[cfg(unix)]
    #[test]
    fn pid_cell_tracks_spawn_exit_and_stop() {
        let mut m = manager(&[]);
        let cell = m.pid_cell();
        assert_eq!(*cell.lock().unwrap(), None);
        let child = Command::new("/bin/sleep").arg("30").spawn().unwrap();
        *cell.lock().unwrap() = Some(child.id());
        m.child = Some(child);
        assert!(m.running());
        assert!(cell.lock().unwrap().is_some());
        m.stop();
        assert_eq!(*cell.lock().unwrap(), None, "stop clears the cell");
        // An observed exit clears it too.
        let mut quick = Command::new("/bin/sleep").arg("0").spawn().unwrap();
        let _ = quick.wait();
        m.child = Some(quick);
        *cell.lock().unwrap() = Some(1);
        assert!(!m.running());
        assert_eq!(*cell.lock().unwrap(), None, "observed exit clears the cell");
    }

    #[cfg(unix)]
    #[test]
    fn stop_terminates_a_live_child_and_clears_the_pid_file() {
        let dir = std::env::temp_dir().join(format!("shrike-stop-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("embedding.pid");
        let mut m = manager(&[]);
        m.cfg.pid_file = Some(pid_file.clone());
        m.child = Some(Command::new("/bin/sleep").arg("30").spawn().unwrap());
        m.write_pid_file();
        assert!(m.running());
        assert!(pid_file.exists());
        m.stop();
        assert!(!m.running());
        assert!(!pid_file.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn stop_escalates_to_sigkill_for_a_term_ignoring_child() {
        let mut m = manager(&[]);
        m.child = Some(
            Command::new("/bin/sh")
                .args(["-c", "trap '' TERM; sleep 30"])
                .spawn()
                .unwrap(),
        );
        // Give the shell a beat to install the trap.
        std::thread::sleep(Duration::from_millis(200));
        let started = Instant::now();
        m.stop();
        assert!(!m.running());
        // It waited out SHUTDOWN_TIMEOUT before the kill tier.
        assert!(started.elapsed() >= SHUTDOWN_TIMEOUT);
    }
}
