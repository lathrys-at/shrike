//! S12-1 repro (preserved by lead; rev-S12 worktree reaped, reverted clean).
//! Orphan reap kills an unrelated recycled-PID process: reap_orphan checks
//! pid_alive(recorded) AND port_held(ANY holder) INDEPENDENTLY — never that the
//! recorded PID *is* the port holder. Recycled PID + unrelated loopback-port
//! holder → SIGKILLs a bystander. Contradicts CLAUDE.md "both required so a
//! recycled PID can't kill an unrelated process". RED at fa54f8c.
//! (Windows worse: pid_alive hardcoded true lib.rs:97-102 → guard = port_held alone.)
//! Insert into native/shrike-llama-server/src/lib.rs mod tests.
//! Run: cd native && CARGO_TARGET_DIR=$HOME/.cache/shrike-review-target/s12 cargo test -p shrike-llama-server s12_recycled_pid -- --nocapture
//! Fix: verify the recorded PID actually owns the port (e.g. lsof/proc-net match) before terminate.

#[cfg(unix)]
#[test]
fn s12_recycled_pid_with_unrelated_port_holder_is_wrongly_killed() {
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

    // DESIRED: U survives (it does not hold our port → not an orphan).
    // PREDICTED FAILURE: U SIGKILLed (port_held saw H, pid_alive saw U).
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
