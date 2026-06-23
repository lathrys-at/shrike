//! Port-ownership orphan reaping — the safety-critical core.
//!
//! A managed subprocess survives a parent SIGKILL, so a later start must be
//! able to reap an orphan left holding our port by a prior unclean shutdown.
//! The reap is gated on a **dual signal**: the recorded PID is terminated
//! **only when it is BOTH still alive AND still the process LISTENing on our
//! port** — both required, so a recycled PID (now some unrelated live process)
//! can never take a bystander down with it. This module owns that gate and the
//! port→PID attribution it rests on; [`Supervisor`](crate::Supervisor) calls
//! into it but the kill decision lives here.
//!
//! The attribution is **positive-only**: a mechanism may place a PID in
//! the owner set *only* when it has proven that PID owns a LISTEN socket on the
//! port. "Ran but found nothing" is `Some(empty)` (authoritative: no owner);
//! "could not run at all" is `None` (ownership unprovable → do not kill). A
//! fallback that cannot prove ownership (permission error, unparseable, tool
//! absent) falls through to `None`/omit-the-PID — never a guess.

use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use crate::which;

/// Whether `pid` is currently alive. `kill(pid, 0)` on unix: 0 = exists,
/// `EPERM` = exists but not ours. Off unix there is no cheap existence probe,
/// so the port half of the dual signal carries the reap decision alone.
pub(crate) fn pid_alive(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
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

/// Send SIGTERM (`hard=false`) or SIGKILL (`hard=true`) to a raw PID. Off unix
/// there is no graceful tier from outside the handle, so both map to a forceful
/// `taskkill /F` (mirroring Python's `os.kill` → TerminateProcess).
pub(crate) fn terminate_raw(pid: i64, hard: bool) {
    #[cfg(unix)]
    {
        let sig = if hard { libc::SIGKILL } else { libc::SIGTERM };
        unsafe {
            libc::kill(pid as libc::pid_t, sig);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = hard;
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// "Is the port available to bind right now" — a `TcpListener::bind` probe
/// (instant; a connect probe's 500ms timeout would dominate any wait loop built
/// on it). The probe listener is dropped immediately.
pub(crate) fn port_bindable(host: &str, port: u16) -> bool {
    TcpListener::bind((host, port)).is_ok()
}

/// Something else holds the port — `EADDRINUSE` specifically, so an unrelated
/// bind failure (e.g. an unresolvable host) never reads as "held". Test-only:
/// the reap gate verifies *which PID* owns the port
/// ([`pid_owns_port`]), never the bare "someone holds it" signal this gave
/// (that signal, paired with a recycled PID, was what killed bystanders).
#[cfg(test)]
pub(crate) fn port_held(host: &str, port: u16) -> bool {
    matches!(
        TcpListener::bind((host, port)),
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse
    )
}

/// The PIDs currently LISTENing on `port`, or `None` if ownership could not be
/// established (no mechanism could run). The reap gate treats `None` as "do not
/// kill" — we never terminate a PID we cannot prove owns our port, so a recycled
/// PID can't take a bystander down with it.
///
/// Invariant (safety-critical — this feeds a kill gate): every mechanism
/// here may put a PID in the returned set *only* when it has positively proven
/// that PID owns a LISTEN socket on `port`. "Ran but found nothing" is
/// `Some(empty)` (authoritative: no owner); "could not run at all" is `None`
/// (ownership unprovable → do not kill). A fallback that cannot prove ownership
/// (permission error, unparseable, tool absent) must fall through to
/// `None`/omit-the-PID — never guess. A bind to all interfaces means the host
/// part of our address is immaterial to ownership — we match on the numeric port
/// alone, which is what we are about to bind.
///
/// Unix fallback chain, each preserving the invariant:
///   1. `lsof`, resolved robustly (PATH, then well-known absolute paths) — the
///      sandbox strips `/usr/sbin` from `PATH` but still permits exec'ing the
///      absolute binary, so this restores macOS coverage. An lsof that *ran* is
///      authoritative (`Some(parsed pids)`), even when empty.
///   2. (Linux only) `/proc/net/tcp{,6}` + `/proc/<pid>/fd` scan, used *only*
///      when no lsof could run — a dependency-free, positive port→inode→pid
///      attribution for lsof-less containers.
///
/// Kept callable standalone (`pub`) for diagnostics, not just the reap gate.
pub fn port_owner_pids(port: u16) -> Option<Vec<i64>> {
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
            proc_net_owner_pids(port)
        }
        // Step 3: no mechanism could run (no lsof; not Linux, so no /proc).
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }
    #[cfg(not(unix))]
    {
        // `netstat -ano` lists "Proto Local Foreign State PID"; take the PID of
        // any LISTENING row whose local address ends in `:PORT`. This is what
        // lets the Windows guard escape the `pid_alive == true` collapse:
        // ownership is established by the port→PID match, not a port-held OR a
        // hardcoded-alive signal.
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

/// Run `lsof` (resolved robustly) against `port`, or `None` if no resolution of
/// it could be spawned. The bazel `darwin-sandbox` strips `/usr/sbin` from
/// `PATH`, so `which("lsof")` fails even though `/usr/sbin/lsof` exists and is
/// exec'able by absolute path — try PATH first, then the well-known absolute
/// locations. A spawn that *ran* is authoritative (`Some`, even empty); a
/// non-zero exit when nothing matches is a legitimate "no owner", not a failure.
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

/// Linux `/proc`-only fallback: collect the PIDs LISTENing on `port` by
/// parsing `/proc/net/tcp{,6}` for the listening socket inodes, then scanning
/// `/proc/<pid>/fd/*` symlinks for `socket:[<inode>]`. Pure-Rust, dependency-
/// free, and *positive* (port → inode → pid). Returns `Some(pids)` when
/// `/proc/net/tcp{,6}` was readable (`Some(empty)` = readable, no LISTEN owner)
/// and `None` when neither file could be read (ownership unprovable → no kill).
///
/// Conservative on failure: a `/proc/<pid>/fd` scan that hits a
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
    // for a symlink target of `socket:[<inode>]`. A permission error on a given
    // pid's fd dir omits that pid (conservative — never misattribute).
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

/// Parse `/proc/net/tcp` (or `/proc/net/tcp6`) text, returning the socket inodes
/// of rows in the LISTEN state whose local-address port equals `port`. Split
/// from the filesystem so it is unit-testable over fixture text (the
/// safety-critical correctness).
///
/// Row layout (whitespace-split, after the header line): field[1] =
/// `local_address` as `HEXIP:HEXPORT` (the port is the hex after the final `:`;
/// the IP is 8 hex digits for tcp, 32 for tcp6 — immaterial, we match on the
/// port alone like the rest of this module); field[3] = `st`, the connection
/// state (`0A` = TCP_LISTEN); field[9] = `inode`. A row missing any field, with
/// a non-hex port, or not in LISTEN state is ignored — never guessed at.
/// Port match is exact: `0x1538` must not match `0x0538` or `0x15380`, which
/// equality of the parsed `u16` gives for free.
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
            // local_address = HEXIP:HEXPORT — take the hex port after the final
            // ':' and parse it. A missing ':' or non-hex port is an unparseable
            // row → ignored (never a guess).
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

/// Whether `pid` is provably the (a) process LISTENing on `port`. Returns false
/// both when some *other* process owns the port (the recycled-PID +
/// unrelated-holder case) and when ownership cannot be determined at all — the
/// kill is gated on a positive match, never on an absent signal.
///
/// Kept callable standalone (`pub`) for diagnostics, not just the reap gate.
pub fn pid_owns_port(pid: i64, port: u16) -> bool {
    matches!(port_owner_pids(port), Some(owners) if owners.contains(&pid))
}

/// Poll-wait for a PID to die. This is the kill confirmation: `pid_alive` going
/// false, never the port — port state is not process identity (an unrelated
/// process can grab the freed port mid-window). Only meaningful for a *non-child*
/// PID (a prior process's orphan, reparented to init and reaped there); our own
/// child would zombie until waited.
pub(crate) fn wait_pid_dead(pid: i64, timeout: Duration) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    // The /proc/net/tcp parser is the safety-critical correctness of the
    // Linux fallback: a misparsed port = a wrong inode = (after fd-scan) a wrong
    // PID killed. Cover it thoroughly over fixture text — pure function, no real
    // /proc needed. Real-kernel row format verified against net/ipv4/tcp_ipv4.c
    // get_tcp4_sock (cols 1/3/9, st 0A = LISTEN, port is the ntohs'd hex after
    // the final ':', no byte-swap).
    #[cfg(target_os = "linux")]
    #[test]
    fn proc_net_parser_extracts_only_listen_rows_on_the_exact_port() {
        // Port 0x1538 = 5432. A realistic /proc/net/tcp with a header, a LISTEN
        // row on our port (inode 23456), an ESTABLISHED row on the same port
        // (must be ignored — not a listener), and a LISTEN row on a different
        // port (must be ignored — wrong port).
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
        // tcp6 has a 32-hex-digit local IP; the port is still the hex after the
        // final ':'. Mixed with malformed rows that must all be skipped, never
        // guessed: too-few columns, a non-hex port, an empty line, and the
        // header. Only the well-formed tcp6 LISTEN row on 5432 survives.
        let text = "\
  sl  local_address                         rem_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000000000000000000000000000:1538 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 54321 1 0000000000000000 100 0 0 10 0
   1: short row that does not have enough columns
   2: 0100007F:ZZZZ 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 99999 1 0 100 0 0 10 0

   3: 0100007F:1538 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 notanumber 1 0 100 0 0 10 0
";
        // The tcp6 LISTEN row (inode 54321) matches; the malformed rows and the
        // un-parseable-inode row are all dropped.
        assert_eq!(parse_proc_net_listen_inodes(text, 5432), vec![54321]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn proc_net_parser_empty_or_header_only_input_yields_nothing() {
        // Empty input and a header-only file are both "readable, no owner" — an
        // empty vec, never a panic, never a spurious inode.
        assert!(parse_proc_net_listen_inodes("", 5432).is_empty());
        assert!(parse_proc_net_listen_inodes(
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n",
            5432
        )
        .is_empty());
    }

    #[test]
    fn bind_probe_distinguishes_free_held_and_unresolvable() {
        // Bind port 0 so the OS hands us a port nothing else holds — a static
        // port races concurrent tests (and anything else on the machine) into
        // flaky free/held assertions.
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        // Held: a live listener → not bindable, held (EADDRINUSE).
        assert!(!port_bindable("127.0.0.1", port));
        assert!(port_held("127.0.0.1", port));
        // Released: becomes bindable again — but never assert immediately, a
        // just-closed listener can be transiently unbindable (close→rebind
        // race), so retry briefly like `wait_port_bindable` does.
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

    // The dual-signal invariant, pinned at the attribution layer (independent of
    // any Supervisor policy): `pid_owns_port` is a POSITIVE match. A live PID
    // that does NOT own the port (the recycled-PID case) reads false, so the
    // reap gate that ANDs alive + owns-port can never kill it.
    #[cfg(unix)]
    #[test]
    fn pid_owns_port_is_false_for_a_live_non_holder() {
        // A live process (ourselves) that does not LISTEN on a freshly-freed
        // port must read as "does not own" — the bystander-survival guarantee.
        let probe = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let free_port = probe.local_addr().unwrap().port();
        drop(probe); // nothing LISTENs here now
        let self_pid = std::process::id() as i64;
        assert!(
            !pid_owns_port(self_pid, free_port),
            "a live PID that does not hold the port must not read as owner — \
             this is the recycled-PID bystander guard"
        );
    }

    // ---- Adversarial reaper boundary tests (#744) ----

    // `pid_owns_port` is gated on a positive port→PID match: a non-positive PID
    // can never own a port (it is not a real PID), so the kill gate that ANDs on
    // it can never fire for one. Pins the guard against a garbage recorded PID.
    #[cfg(unix)]
    #[test]
    fn pid_owns_port_is_false_for_nonpositive_pids() {
        // Bind a real listener so SOMETHING owns the port — proving the false is
        // the PID guard, not merely "no owner".
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!pid_owns_port(0, port), "PID 0 never owns a port");
        assert!(!pid_owns_port(-1, port), "a negative PID never owns a port");
        drop(listener);
    }

    // `pid_alive` rejects non-positive PIDs without ever calling kill() — a
    // recorded "0" or negative must read dead so the reaper short-circuits.
    #[cfg(unix)]
    #[test]
    fn pid_alive_is_false_for_nonpositive_pids() {
        assert!(!pid_alive(0));
        assert!(!pid_alive(-1));
        assert!(!pid_alive(i64::MIN));
    }

    // On a port nothing LISTENs on, ownership is authoritatively "no owner":
    // `pid_owns_port` reads false for ANY pid. Where the platform can introspect
    // ownership at all (lsof / Linux /proc), `port_owner_pids` is `Some(empty)`
    // — never `None` (which would mean "unprovable") and never a guess.
    #[cfg(unix)]
    #[test]
    fn no_owner_on_a_free_port_yields_empty_not_a_guess() {
        let probe = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe); // free the port — nothing LISTENs now
        let self_pid = std::process::id() as i64;
        // Capability gate: only assert Some(empty) where ownership is observable
        // here (a positive self-ownership probe), mirroring the existing
        // positive-reap test's skip discipline so an lsof-less sandbox doesn't
        // fail on an unprovable-by-design platform.
        let observable = {
            let l = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let p = l.local_addr().unwrap().port();
            let ok = pid_owns_port(self_pid, p);
            drop(l);
            ok
        };
        if observable {
            assert_eq!(
                port_owner_pids(port),
                Some(Vec::new()),
                "a free port is authoritatively 'no owner', never None or a guess"
            );
        }
        assert!(!pid_owns_port(self_pid, port), "no PID owns a free port");
    }

    // `wait_pid_dead` returns true immediately for an already-dead PID — no spin
    // to the timeout. Use a high, almost-certainly-unused PID number.
    #[cfg(unix)]
    #[test]
    fn wait_pid_dead_returns_immediately_for_a_dead_pid() {
        let started = Instant::now();
        // 0x7FFF_FFFF is far above any realistic live PID; even if taken, the
        // function still returns a bool — we assert on the dead case via a PID we
        // know is invalid (0 reads dead through pid_alive's guard).
        assert!(wait_pid_dead(0, Duration::from_secs(5)));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "a dead PID must not spin to the timeout"
        );
    }

    // `port_held` distinguishes EADDRINUSE from other bind errors only — pinned
    // here for the unresolvable-host case independently of the bind-probe test:
    // a bind failure that is NOT AddrInUse must never read as "held" (that false
    // positive, paired with a recycled PID, was the original mis-kill).
    #[test]
    fn port_held_only_true_for_address_in_use() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(port_held("127.0.0.1", port), "a live listener reads held");
        assert!(
            !port_held("host.invalid.shrike-test-xyz", port),
            "a non-EADDRINUSE bind failure must not read as held"
        );
        drop(listener);
    }
}
