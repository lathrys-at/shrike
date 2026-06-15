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
/// 50ms, not 250 (#426): a localhost health GET costs ~1ms, and every service
/// start rounds up to one poll quantum — a small model loads faster than a
/// single 250ms tick, so the coarser interval was pure added boot latency
/// (felt acutely by the embedding test suites, which boot servers per fixture).
pub const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(50);
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
    "--mmproj",
];
/// Of the reserved flags, those that consume a following value token (so a
/// rejected `--host 0.0.0.0` drops the value too, not just the flag).
pub const RESERVED_VALUE_FLAGS: &[&str] = &["--model", "-m", "--host", "--port", "--mmproj"];

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
    /// Serve embeddings (the default) or chat — a describe/vision server
    /// (#433) is a *chat* server and must not pass `--embeddings`.
    embeddings: bool,
    /// Multimodal projector(s) (`--mmproj`, repeatable): one for a vision
    /// chat server (#433); one per modality for a multimodal *embeddings*
    /// server (#501 — jina-v5-omni ships separate vision/audio mmprojs,
    /// loaded together for both).
    mmprojs: Vec<String>,
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
/// unrelated bind failure (e.g. an unresolvable host) never reads as "held".
/// Test-only since #594: the reap gate verifies *which PID* owns the port
/// ([`pid_owns_port`]), never the bare "someone holds it" signal this gave
/// (that signal, paired with a recycled PID, was what killed bystanders).
#[cfg(test)]
fn port_held(host: &str, port: u16) -> bool {
    matches!(
        TcpListener::bind((host, port)),
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse
    )
}

/// The PIDs currently LISTENing on `port`, or `None` if ownership could not
/// be established (no mechanism could run). The reap gate treats `None` as
/// "do not kill" — we never terminate a PID we cannot prove owns our port, so
/// a recycled PID can't take a bystander down with it.
///
/// #594 invariant (safety-critical — this feeds a kill gate): every mechanism
/// here may put a PID in the returned set *only* when it has positively proven
/// that PID owns a LISTEN socket on `port`. "Ran but found nothing" is
/// `Some(empty)` (authoritative: no owner); "could not run at all" is `None`
/// (ownership unprovable → do not kill). A fallback that cannot prove
/// ownership (permission error, unparseable, tool absent) must fall through to
/// `None`/omit-the-PID — never guess. A bind to all interfaces means the host
/// part of our address is immaterial to ownership — we match on the numeric
/// port alone, which is what we are about to bind.
///
/// #654 unix fallback chain, each preserving the invariant:
///   1. `lsof`, resolved robustly (PATH, then well-known absolute paths) — the
///      sandbox strips `/usr/sbin` from `PATH` but still permits exec'ing the
///      absolute binary, so this restores macOS coverage. An lsof that *ran*
///      is authoritative (`Some(parsed pids)`), even when empty.
///   2. (Linux only) `/proc/net/tcp{,6}` + `/proc/<pid>/fd` scan, used *only*
///      when no lsof could run — a dependency-free, positive port→inode→pid
///      attribution for lsof-less containers.
fn port_owner_pids(port: u16) -> Option<Vec<i64>> {
    #[cfg(unix)]
    {
        // Step 1: lsof, if any resolution of it can be spawned. A spawn that
        // ran is authoritative — do NOT then also scan /proc (step 3 below).
        if let Some(pids) = lsof_owner_pids(port) {
            return Some(pids);
        }
        // Step 2 (Linux): no lsof could run — fall back to /proc. Returns
        // Some(pids) when /proc/net/tcp{,6} was readable (Some(empty) = "ran,
        // found no LISTEN owner"), None when it could not be read at all.
        #[cfg(target_os = "linux")]
        {
            return proc_net_owner_pids(port);
        }
        // Step 3: no mechanism could run (no lsof; not Linux, so no /proc).
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }
    #[cfg(not(unix))]
    {
        // `netstat -ano` lists "Proto Local Foreign State PID"; take the PID
        // of any LISTENING row whose local address ends in `:PORT`. This is
        // what lets the Windows guard escape the `pid_alive == true` collapse:
        // ownership is established by the port→PID match, not a port-held OR
        // a hardcoded-alive signal.
        let out = Command::new("netstat")
            .args(["-ano", "-p", "TCP"])
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        let needle = format!(":{port}");
        let mut pids = Vec::new();
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let cols: Vec<&str> = line.split_whitespace().collect();
            // Proto Local-Address Foreign-Address State PID
            if cols.len() >= 5
                && cols[0].eq_ignore_ascii_case("TCP")
                && cols[3].eq_ignore_ascii_case("LISTENING")
                && cols[1].ends_with(&needle)
            {
                if let Ok(pid) = cols[4].parse::<i64>() {
                    pids.push(pid);
                }
            }
        }
        Some(pids)
    }
}

/// Run `lsof` (resolved robustly) against `port`, or `None` if no resolution
/// of it could be spawned. #654: the bazel `darwin-sandbox` strips `/usr/sbin`
/// from `PATH`, so `which("lsof")` fails even though `/usr/sbin/lsof` exists
/// and is exec'able by absolute path — try PATH first, then the well-known
/// absolute locations. A spawn that *ran* is authoritative (`Some`, even
/// empty); a non-zero exit when nothing matches is a legitimate "no owner",
/// not a failure (#594).
#[cfg(unix)]
fn lsof_owner_pids(port: u16) -> Option<Vec<i64>> {
    // PATH resolution first (the common case), then well-known absolute paths
    // for the PATH-stripped sandbox / minimal-env case. macOS ships lsof at
    // /usr/sbin; Linux distros put it in /usr/bin or /bin.
    let resolved = which("lsof");
    let candidates = [
        resolved.as_deref(),
        Some("/usr/sbin/lsof"),
        Some("/usr/bin/lsof"),
        Some("/bin/lsof"),
    ];
    // -nP: no name/port resolution (fast, unambiguous); -t: terse, one PID per
    // line; -sTCP:LISTEN: only the listener, never a transient client connected
    // TO the port.
    for bin in candidates.into_iter().flatten() {
        match Command::new(bin)
            .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-t"])
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
        {
            Ok(out) => {
                // It spawned — authoritative. Parse the PID-per-line stdout
                // (empty when nothing matches; lsof exits non-zero then, which
                // is "no owner", not a failure to run).
                let pids = String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .filter_map(|l| l.trim().parse::<i64>().ok())
                    .collect();
                return Some(pids);
            }
            // NotFound / permission-on-exec: try the next candidate path.
            Err(_) => continue,
        }
    }
    None
}

/// Linux `/proc`-only fallback (#654): collect the PIDs LISTENing on `port` by
/// parsing `/proc/net/tcp{,6}` for the listening socket inodes, then scanning
/// `/proc/<pid>/fd/*` symlinks for `socket:[<inode>]`. Pure-Rust, dependency-
/// free, and *positive* (port → inode → pid). Returns `Some(pids)` when
/// `/proc/net/tcp{,6}` was readable (`Some(empty)` = readable, no LISTEN owner)
/// and `None` when neither file could be read (ownership unprovable → no kill).
///
/// Conservative on failure (#594): a `/proc/<pid>/fd` scan that hits a
/// permission error just *omits* that PID — failing to attribute means "don't
/// reap" (safe), never "reap the wrong one". Our own prior-uid orphan's fds are
/// readable, so a real orphan is still found.
#[cfg(target_os = "linux")]
fn proc_net_owner_pids(port: u16) -> Option<Vec<i64>> {
    // Gather the listening-socket inodes from both tcp and tcp6. Read both
    // before deciding readability: Some iff *at least one* file was readable
    // (a kernel may lack ipv6); None only when neither could be read at all.
    let mut inodes: Vec<u64> = Vec::new();
    let mut any_readable = false;
    for path in ["/proc/net/tcp", "/proc/net/tcp6"] {
        if let Ok(text) = std::fs::read_to_string(path) {
            any_readable = true;
            inodes.extend(parse_proc_net_listen_inodes(&text, port));
        }
    }
    if !any_readable {
        return None;
    }
    if inodes.is_empty() {
        // /proc/net/tcp{,6} was readable but nothing LISTENs on `port` —
        // authoritative "no owner".
        return Some(Vec::new());
    }

    // Map socket inodes → owning PIDs by scanning each numeric /proc/<pid>/fd
    // for a symlink target of `socket:[<inode>]`. A permission error on a
    // given pid's fd dir omits that pid (conservative — never misattribute).
    let mut targets: Vec<String> = inodes.iter().map(|i| format!("socket:[{i}]")).collect();
    targets.sort_unstable();
    targets.dedup();
    let mut pids: Vec<i64> = Vec::new();
    let Ok(proc_dir) = std::fs::read_dir("/proc") else {
        // /proc/net was readable but /proc itself isn't enumerable — we cannot
        // attribute the inodes to PIDs. Omit (no positive ownership), which is
        // Some(empty) here: readable-but-unattributable is "do not kill".
        return Some(Vec::new());
    };
    for entry in proc_dir.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // Numeric directory names are PIDs; skip everything else.
        let Ok(pid) = name.parse::<i64>() else {
            continue;
        };
        let fd_dir = entry.path().join("fd");
        let Ok(fds) = std::fs::read_dir(&fd_dir) else {
            // Permission error / process gone — omit this pid (conservative).
            continue;
        };
        for fd in fds.flatten() {
            // Each fd is a symlink; its target is `socket:[<inode>]` for a
            // socket fd. read_link, not metadata, so we compare the link text.
            if let Ok(link) = std::fs::read_link(fd.path()) {
                let link = link.to_string_lossy();
                if targets.binary_search(&link.to_string()).is_ok() {
                    pids.push(pid);
                    break; // one matching fd is enough to attribute the pid
                }
            }
        }
    }
    pids.sort_unstable();
    pids.dedup();
    Some(pids)
}

/// Parse `/proc/net/tcp` (or `/proc/net/tcp6`) text, returning the socket
/// inodes of rows in the LISTEN state whose local-address port equals `port`.
/// Split from the filesystem so it is unit-testable over fixture text (the
/// safety-critical correctness; #654).
///
/// Row layout (whitespace-split, after the header line): field[1] =
/// `local_address` as `HEXIP:HEXPORT` (the port is the hex after the final
/// `:`; the IP is 8 hex digits for tcp, 32 for tcp6 — immaterial, we match on
/// the port alone like the rest of this module); field[3] = `st`, the
/// connection state (`0A` = TCP_LISTEN); field[9] = `inode`. A row missing any
/// field, with a non-hex port, or not in LISTEN state is ignored — never
/// guessed at (#594). Port match is exact: `0x1538` must not match `0x0538`
/// or `0x15380`, which equality of the parsed `u16` gives for free.
#[cfg(target_os = "linux")]
fn parse_proc_net_listen_inodes(text: &str, port: u16) -> Vec<u64> {
    // The first line is the column header (`sl  local_address ...`); it has no
    // leading numeric `sl:` so it parses to nothing anyway, but skip it
    // explicitly for clarity.
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let cols: Vec<&str> = line.split_whitespace().collect();
            // Need at least through the inode column (index 9).
            if cols.len() <= 9 {
                return None;
            }
            // State must be LISTEN (0A). Compare case-insensitively — the
            // kernel emits uppercase, but don't depend on it.
            if !cols[3].eq_ignore_ascii_case("0A") {
                return None;
            }
            // local_address = HEXIP:HEXPORT — take the hex port after the
            // final ':' and parse it. A missing ':' or non-hex port is an
            // unparseable row → ignored (never a guess).
            let (_ip, hex_port) = cols[1].rsplit_once(':')?;
            let row_port = u16::from_str_radix(hex_port, 16).ok()?;
            if row_port != port {
                return None;
            }
            // inode is a decimal column; an unparseable inode drops the row.
            cols[9].parse::<u64>().ok()
        })
        .collect()
}

/// Whether `pid` is provably the (a) process LISTENing on `port`. Returns
/// false both when some *other* process owns the port (the recycled-PID +
/// unrelated-holder case) and when ownership cannot be determined at all —
/// the kill is gated on a positive match, never on an absent signal.
fn pid_owns_port(pid: i64, port: u16) -> bool {
    matches!(port_owner_pids(port), Some(owners) if owners.contains(&pid))
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
            embeddings: true,
            mmprojs: Vec::new(),
            pid_cell: std::sync::Arc::default(),
        }
    }

    /// Reconfigure as a *chat* server (no `--embeddings`) with an optional
    /// multimodal projector — the shape a remote-describe deployment runs
    /// (#433): `llama-server -m model.gguf --mmproj proj.gguf`. A builder on
    /// the manager (not a config field) so the existing exhaustive config
    /// constructors stay valid. Vision models want a generous
    /// `context_size` — image tokens are expensive.
    pub fn chat_mode(mut self, mmproj: Option<String>) -> Self {
        self.embeddings = false;
        self.mmprojs = mmproj.into_iter().collect();
        self
    }

    /// Load multimodal projector(s) on an *embeddings* server (#501): the
    /// shape behind a multimodal `embedders:` entry served by the managed
    /// llama-server. Repeatable because per-modality mmprojs load together
    /// (vision + audio for an omni model). Keeps `--embeddings` on.
    pub fn with_mmprojs(mut self, mmprojs: Vec<String>) -> Self {
        self.mmprojs = mmprojs;
        self
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
        ];
        if self.embeddings {
            cmd.push("--embeddings".into());
        }
        for mmproj in &self.mmprojs {
            cmd.extend(["--mmproj".into(), mmproj.clone()]);
        }
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

    /// Kill a llama-server left over from a prior unclean shutdown. The
    /// orphan is the recorded PID **only if that PID is itself the process
    /// LISTENing on our port** — checked with [`pid_owns_port`], not the old
    /// independent `pid_alive && port_held` pair. That pair could kill an
    /// unrelated process: a recycled PID (now some other live process) plus
    /// *any* unrelated holder of our port satisfied both signals without the
    /// PID ever being the holder. Requiring the PID to own the port closes
    /// that, and fails safe (no positive ownership → no kill) when ownership
    /// can't be established. Private: it clears the PID file as a side effect,
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
        // `pid_owns_port` already implies the PID is alive (it holds a
        // socket); the cheap `pid_alive` short-circuits the lsof spawn for an
        // obviously-dead recorded PID.
        if pid_alive(pid) && pid_owns_port(pid, self.cfg.port) {
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

    /// Poll until the port binds. A just-released port can be transiently
    /// unbindable (TIME_WAIT / close→rebind races), so this can spin
    /// briefly even after the holder's death is already confirmed.
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
    fn chat_mode_drops_embeddings_and_emits_mmproj() {
        let m = manager(&[]).chat_mode(Some("/models/proj.gguf".into()));
        let cmd = m.build_command("/bin/llama-server");
        assert!(!cmd.contains(&"--embeddings".to_string()), "{cmd:?}");
        let i = cmd
            .iter()
            .position(|t| t == "--mmproj")
            .expect("mmproj emitted");
        assert_eq!(cmd[i + 1], "/models/proj.gguf");
        // Default mode is unchanged: embeddings on, no projector.
        let cmd = manager(&[]).build_command("/bin/llama-server");
        assert!(cmd.contains(&"--embeddings".to_string()));
        assert!(!cmd.contains(&"--mmproj".to_string()));
    }

    #[test]
    fn embeddings_mode_emits_repeatable_mmprojs() {
        // The #501 omni shape: per-modality projectors load together on an
        // EMBEDDINGS server — `--embeddings` stays on, one `--mmproj` each,
        // in declaration order.
        let m = manager(&[]).with_mmprojs(vec![
            "/models/vision.mmproj.gguf".into(),
            "/models/audio.mmproj.gguf".into(),
        ]);
        let cmd = m.build_command("/bin/llama-server");
        assert!(cmd.contains(&"--embeddings".to_string()), "{cmd:?}");
        let positions: Vec<usize> = cmd
            .iter()
            .enumerate()
            .filter(|(_, t)| *t == "--mmproj")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(positions.len(), 2, "{cmd:?}");
        assert_eq!(cmd[positions[0] + 1], "/models/vision.mmproj.gguf");
        assert_eq!(cmd[positions[1] + 1], "/models/audio.mmproj.gguf");
    }

    #[test]
    fn mmproj_is_reserved_from_passthrough() {
        // The typed chat_mode owns the flag — a passthrough --mmproj (and
        // its value) is stripped like any reserved flag.
        let m = manager(&["--mmproj /evil/proj.gguf --flash-attn"]);
        assert_eq!(m.passthrough_tokens(false), vec!["--flash-attn"]);
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

    #[cfg(unix)]
    #[test]
    fn s12_recycled_pid_with_unrelated_port_holder_is_wrongly_killed() {
        // #594 (audit S12-1): a recycled PID recorded for reaping, plus an
        // UNRELATED process holding our port — the old `pid_alive(recorded)
        // && port_held(ANY holder)` pair killed the bystander because it
        // never asked whether the recorded PID *is* the holder. The unrelated
        // process (U) must survive: it does not own our port → not an orphan.
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port(); // H holds this port

        let dir = std::env::temp_dir().join(format!("shrike-s12-wrongkill-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("embedding.pid");
        // U: an unrelated live process whose PID landed in our reap record.
        let mut victim = Command::new("/bin/sleep").arg("30").spawn().unwrap();
        std::fs::write(&pid_file, victim.id().to_string()).unwrap();

        let mut m = manager(&[]);
        m.cfg.port = port; // reap against the port H holds
        m.cfg.pid_file = Some(pid_file.clone());

        m.reap_orphan();

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
        // *detached* (reparented-to-init) process that genuinely LISTENs on
        // our port — a real orphan — and whose PID is recorded must still be
        // terminated. Detach via `sh -c '... & echo $!'` like
        // terminate_pid_confirms_death_via_pid_not_port, so the SIGTERM'd
        // process is reaped by init rather than zombied (a direct child would
        // linger as a zombie that `kill(pid, 0)` still reports alive).
        // Capability gate (#652): this test verifies a *positive* reap, which
        // needs to confirm ownership via `port_owner_pids` (lsof on unix). The
        // bazel darwin-sandbox provides no `lsof`, so ownership can never be
        // established there and the setup precondition below could never hold.
        // Probe the capability *positively* — bind an in-process listener and
        // ask whether we can observe our OWN PID owning it. If we cannot, the
        // platform can't introspect port ownership here, so skip cleanly (a
        // returning Rust test is reported PASSED). Keying the skip on this
        // positive incapability — not on catching the would-be setup failure —
        // keeps a genuine reap regression FAILING rather than silently skipped:
        // wherever lsof exists (dev machines, CI Linux) the probe passes and the
        // full reap is exercised. The production reap is unaffected — it already
        // treats "ownership unprovable" as "never kill" (fail-open, gated).
        {
            let self_listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
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
        let pid_file = dir.join("embedding.pid");
        // Pick a free port, then hand it to the detached listener.
        let probe = TcpListener::bind(("127.0.0.1", 0)).unwrap();
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

        let mut m = manager(&[]);
        m.cfg.port = port;
        m.cfg.pid_file = Some(pid_file.clone());
        m.reap_orphan();

        // The genuine orphan was terminated and the record cleared.
        let dead = !pid_alive(pid);
        if !dead {
            terminate_raw(pid, true); // belt-and-suspenders cleanup
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(dead, "a real orphan holding our port must still be reaped");
        assert!(!pid_file.exists(), "the record is cleared after the reap");
    }

    #[test]
    fn bind_probe_distinguishes_free_held_and_unresolvable() {
        // Bind port 0 so the OS hands us a port nothing else holds — a
        // static port races concurrent tests (and anything else on the
        // machine) into flaky free/held assertions.
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        // Held: a live listener → not bindable, held (EADDRINUSE).
        assert!(!port_bindable("127.0.0.1", port));
        assert!(port_held("127.0.0.1", port));
        // Released: becomes bindable again — but never assert immediately,
        // a just-closed listener can be transiently unbindable (close→
        // rebind race), so retry briefly like `wait_port_bindable` does.
        drop(listener);
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut freed = port_bindable("127.0.0.1", port);
        while !freed && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
            freed = port_bindable("127.0.0.1", port);
        }
        assert!(freed, "released port never became bindable");
        // An unresolvable host is NOT "held" — a bind failure that isn't
        // EADDRINUSE must never corroborate a kill.
        assert!(!port_held("host.invalid.shrike-test", port));
    }

    // The /proc/net/tcp parser is the safety-critical correctness of the #654
    // Linux fallback: a misparsed port = a wrong inode = (after fd-scan) a
    // wrong PID killed. Cover it thoroughly over fixture text — pure function,
    // no real /proc needed. Real-kernel row format verified against
    // net/ipv4/tcp_ipv4.c get_tcp4_sock (cols 1/3/9, st 0A = LISTEN, port is
    // the ntohs'd hex after the final ':', no byte-swap).
    #[cfg(target_os = "linux")]
    #[test]
    fn proc_net_parser_extracts_only_listen_rows_on_the_exact_port() {
        // Port 0x1538 = 5432. A realistic /proc/net/tcp with a header, a
        // LISTEN row on our port (inode 23456), an ESTABLISHED row on the same
        // port (must be ignored — not a listener), and a LISTEN row on a
        // different port (must be ignored — wrong port).
        let text = "\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:1538 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 23456 1 0000000000000000 100 0 0 10 0
   1: 0100007F:1538 0100007F:9999 01 00000000:00000000 00:00000000 00000000     0        0 77777 1 0000000000000000 20 4 30 10 -1
   2: 00000000:0050 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 88888 1 0000000000000000 100 0 0 10 0
";
        // Only the LISTEN row on port 5432 contributes its inode.
        assert_eq!(parse_proc_net_listen_inodes(text, 5432), vec![23456]);
        // The different-port LISTEN row (0x0050 = 80) is matched only by 80.
        assert_eq!(parse_proc_net_listen_inodes(text, 80), vec![88888]);
        // A port nothing LISTENs on yields nothing (readable, no owner).
        assert!(parse_proc_net_listen_inodes(text, 1234).is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn proc_net_parser_matches_port_exactly_no_prefix_or_substring() {
        // The exact-boundary guard: a u16 equality, not a hex substring. For
        // target 0x1538, neither 0x0538 (shares trailing hex) nor a row whose
        // port hex is a different value must match. (0x15380 can't appear in a
        // %04X field, but parse it explicitly to a different u16 to be sure.)
        let text = "\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:0538 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 11111 1 0 100 0 0 10 0
   1: 0100007F:5380 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 22222 1 0 100 0 0 10 0
   2: 0100007F:1538 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 33333 1 0 100 0 0 10 0
";
        // 0x1538 = 5432 — only the third row, not 0x0538 (1336) or 0x5380 (21376).
        assert_eq!(parse_proc_net_listen_inodes(text, 0x1538), vec![33333]);
        assert_eq!(parse_proc_net_listen_inodes(text, 0x0538), vec![11111]);
        assert_eq!(parse_proc_net_listen_inodes(text, 0x5380), vec![22222]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn proc_net_parser_handles_tcp6_and_ignores_malformed_rows() {
        // tcp6 has a 32-hex-digit local IP; the port is still the hex after
        // the final ':'. Mixed with malformed rows that must all be skipped,
        // never guessed: too-few columns, a non-hex port, an empty line, and
        // the header. Only the well-formed tcp6 LISTEN row on 5432 survives.
        let text = "\
  sl  local_address                         rem_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000000000000000000000000000:1538 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 54321 1 0000000000000000 100 0 0 10 0
   1: short row that does not have enough columns
   2: 0100007F:ZZZZ 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 99999 1 0 100 0 0 10 0

   3: 0100007F:1538 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 notanumber 1 0 100 0 0 10 0
";
        // The tcp6 LISTEN row (inode 54321) matches; the malformed rows and
        // the un-parseable-inode row are all dropped.
        assert_eq!(parse_proc_net_listen_inodes(text, 5432), vec![54321]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn proc_net_parser_empty_or_header_only_input_yields_nothing() {
        // Empty input and a header-only file are both "readable, no owner" —
        // an empty vec, never a panic, never a spurious inode.
        assert!(parse_proc_net_listen_inodes("", 5432).is_empty());
        assert!(parse_proc_net_listen_inodes(
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n",
            5432
        )
        .is_empty());
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
