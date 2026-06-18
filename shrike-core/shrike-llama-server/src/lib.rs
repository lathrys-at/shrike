//! llama-server policy (#342 P4b, #710) — the thin `ManagedProcess` impl atop
//! the generic [`shrike_process::Supervisor`]. This crate owns only what is
//! *llama-specific*: binary resolution (`LLAMA_SERVER_PATH`), the argv
//! vocabulary, the `/health` URL, the model/pooling/mmproj/router shape, and —
//! load-bearing — the reserved-flag security guard. Spawn, health-wait, orphan
//! reaping, and escalating stop are all the supervisor's; the orphan reaper
//! (#594/#654) lives in `shrike-process`, gated on the dual signal.
//!
//! Security boundary preserved verbatim: `--host` is Shrike-owned — the
//! embedding backend is deliberately pinned to loopback (audit §1.1) — so
//! [`LlamaServerConfig`]'s passthrough strips every reserved flag (with its
//! value token for the value-taking ones) before the command is built. The
//! supervisor never sees an unstripped argv.

use std::path::{Path, PathBuf};
use std::time::Duration;

use shrike_error::{NativeError, NativeResult};
use shrike_process::{is_executable, which, ManagedProcess, Supervisor};

// Re-export the shared lifecycle constants so existing `shrike_llama_server::*`
// users keep resolving (the values are the supervisor's now).
pub use shrike_process::{HEALTH_POLL_INTERVAL, HEALTH_TIMEOUT, SHUTDOWN_TIMEOUT, SIGKILL_TIMEOUT};

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
    // Router mode (#566) is Shrike-owned: the model-loading shape (single vs
    // router) is chosen by `ModelSpec`, never smuggled through the passthrough.
    "--models-dir",
    "--models-max",
];
/// Of the reserved flags, those that consume a following value token (so a
/// rejected `--host 0.0.0.0` drops the value too, not just the flag).
pub const RESERVED_VALUE_FLAGS: &[&str] = &[
    "--model",
    "-m",
    "--host",
    "--port",
    "--mmproj",
    "--models-dir",
    "--models-max",
];

/// What the server loads: one model (single-model mode, the default) or a
/// directory of models served by llama.cpp's *router* mode (#566), where the
/// request's `model` field selects among them. The two are mutually exclusive
/// — `--model` and `--models-dir` cannot coexist — so they are an enum, not a
/// bag of optionals.
#[derive(Clone)]
pub enum ModelSpec {
    /// `--model <path>`: a single model loaded at spawn.
    Single(String),
    /// `--models-dir <dir>` (+ optional `--models-max <N>`): router mode (one
    /// process, many models, request-`model`-field routing). Models load lazily
    /// on first request — `/health` is 200 once the router is listening, before
    /// any model loads, so the health-wait is unchanged.
    Router {
        dir: String,
        /// Max models loaded simultaneously (`--models-max`); `None` = the
        /// server default (LRU-evicts beyond it).
        max: Option<u32>,
    },
}

impl std::fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSpec::Single(path) => write!(f, "{path}"),
            ModelSpec::Router { dir, max } => match max {
                Some(n) => write!(f, "router:{dir} (max {n})"),
                None => write!(f, "router:{dir}"),
            },
        }
    }
}

/// llama-server launch configuration. Construct via [`LlamaServerConfig::new`]
/// (model + host + port — the always-required core) and chain the optional
/// setters; [`Default`] is available for `..` spreads in tests. Empty defaults
/// throughout: no binary override, no log dir, server-default context/threads/
/// gpu/pooling, no passthrough, no PID file.
#[derive(Clone, Default)]
pub struct LlamaServerConfig {
    /// Explicit binary override (`--llama-server`); beats env and PATH.
    pub binary: Option<String>,
    pub model: Option<ModelSpec>,
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

impl LlamaServerConfig {
    /// The required core: which model, on which host:port. Everything else
    /// (`binary`, `log_dir`, `context_size`, `threads`, `gpu_layers`, `pooling`,
    /// `extra_args`, `pid_file`) defaults empty and is set with the chainable
    /// `with_*` methods.
    pub fn new(model: ModelSpec, host: impl Into<String>, port: u16) -> Self {
        Self {
            model: Some(model),
            host: host.into(),
            port,
            ..Self::default()
        }
    }

    /// Explicit binary override (beats `LLAMA_SERVER_PATH` and PATH).
    pub fn with_binary(mut self, binary: Option<String>) -> Self {
        self.binary = binary;
        self
    }

    /// Directory for llama-server's `--log-file` and the captured stderr.
    pub fn with_log_dir(mut self, log_dir: Option<PathBuf>) -> Self {
        self.log_dir = log_dir;
        self
    }

    /// `--ctx-size`; `None` keeps the model default.
    pub fn with_context_size(mut self, context_size: Option<u32>) -> Self {
        self.context_size = context_size;
        self
    }

    /// `--threads`; `None` keeps the server default.
    pub fn with_threads(mut self, threads: Option<u32>) -> Self {
        self.threads = threads;
        self
    }

    /// `--gpu-layers`; `None` keeps the server default (CPU).
    pub fn with_gpu_layers(mut self, gpu_layers: Option<i32>) -> Self {
        self.gpu_layers = gpu_layers;
        self
    }

    /// `--pooling` override — required for last-token models whose GGUF omits
    /// the pooling type. `None` uses the model default.
    pub fn with_pooling(mut self, pooling: Option<String>) -> Self {
        self.pooling = pooling;
        self
    }

    /// Raw passthrough entries (each shlex-split at command-build time, then
    /// reserved-flag-stripped).
    pub fn with_extra_args(mut self, extra_args: Vec<String>) -> Self {
        self.extra_args = extra_args;
        self
    }

    /// Where to record the child PID for orphan reaping across an unclean
    /// shutdown.
    pub fn with_pid_file(mut self, pid_file: Option<PathBuf>) -> Self {
        self.pid_file = pid_file;
        self
    }
}

/// The llama-server [`ManagedProcess`] policy: the config plus the
/// embeddings/mmproj shape. Wrapped by [`LlamaServerManager`]; not constructed
/// directly by hosts.
struct LlamaPolicy {
    cfg: LlamaServerConfig,
    /// Serve embeddings (the default) or chat — a describe/vision server (#433)
    /// is a *chat* server and must not pass `--embeddings`.
    embeddings: bool,
    /// Multimodal projector(s) (`--mmproj`, repeatable): one for a vision chat
    /// server (#433); one per modality for a multimodal *embeddings* server
    /// (#501 — jina-v5-omni ships separate vision/audio mmprojs, loaded together
    /// for both).
    mmprojs: Vec<String>,
}

impl LlamaPolicy {
    fn model(&self) -> &ModelSpec {
        self.cfg
            .model
            .as_ref()
            .expect("LlamaServerConfig built without a model — use LlamaServerConfig::new")
    }

    /// Resolve `extra_args` to llama-server tokens, dropping reserved flags —
    /// including a separate value token for value-taking flags (`--host 0.0.0.0`
    /// loses both; the self-contained `--host=0.0.0.0` is one token). `warn` logs
    /// each rejection (the command-build path); the fingerprint reuses this
    /// silently.
    ///
    /// A `Peekable` state machine over the shlex-split tokens: on a reserved
    /// value-taking flag whose value is a *separate following* token, consume
    /// that token too (so it can't survive as a stray positional); the inline
    /// `--flag=value` form is one token and needs no lookahead.
    fn passthrough_tokens(&self, warn: bool) -> Vec<String> {
        let mut raw: Vec<String> = Vec::new();
        for entry in &self.cfg.extra_args {
            if let Some(tokens) = shlex::split(entry) {
                raw.extend(tokens);
            }
        }
        let mut result = Vec::new();
        let mut tokens = raw.into_iter().peekable();
        while let Some(tok) = tokens.next() {
            let flag = tok.split('=').next().unwrap_or(&tok);
            if !RESERVED_FLAGS.contains(&flag) {
                result.push(tok);
                continue;
            }
            if warn {
                tracing::warn!(
                    "Ignoring reserved llama-server flag {flag:?} passed via \
                     --embedding-arg; Shrike controls it (use a typed setting for \
                     vector-affecting flags)."
                );
            }
            // A value-taking reserved flag in the separate-token form (`--host
            // 0.0.0.0`, not `--host=0.0.0.0`) also swallows its following value
            // token, so the value can't survive as a stray positional.
            if RESERVED_VALUE_FLAGS.contains(&flag) && !tok.contains('=') && tokens.peek().is_some()
            {
                tokens.next();
            }
        }
        result
    }

    /// The exact command, Shrike-owned flags first, passthrough last (so a user
    /// can't shadow Shrike's args by ordering; reserved flags are stripped
    /// regardless).
    fn build_command(&self, binary: &str) -> Vec<String> {
        let mut cmd: Vec<String> = vec![binary.to_string()];
        match self.model() {
            ModelSpec::Single(path) => {
                cmd.extend(["--model".into(), path.clone()]);
            }
            ModelSpec::Router { dir, max } => {
                cmd.extend(["--models-dir".into(), dir.clone()]);
                if let Some(n) = max {
                    cmd.extend(["--models-max".into(), n.to_string()]);
                }
            }
        }
        cmd.extend([
            "--host".into(),
            self.cfg.host.clone(),
            "--port".into(),
            self.cfg.port.to_string(),
        ]);
        if self.embeddings {
            cmd.push("--embeddings".into());
        }
        // mmprojs are model-specific (a router serves many models, so no single
        // projector applies globally) — emit them only in single-model mode.
        if matches!(self.model(), ModelSpec::Single(_)) {
            for mmproj in &self.mmprojs {
                cmd.extend(["--mmproj".into(), mmproj.clone()]);
            }
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
            // Override the GGUF's stored pooling type — required for last-token
            // models whose metadata omits it.
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

    /// Locate the llama-server binary: explicit override (validated) >
    /// `LLAMA_SERVER_PATH` (validated) > `PATH`.
    fn find_binary(&self) -> NativeResult<String> {
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
}

impl ManagedProcess for LlamaPolicy {
    fn binary(&self) -> NativeResult<String> {
        self.find_binary()
    }

    fn argv(&self, binary: &str) -> Vec<String> {
        self.build_command(binary)
    }

    fn host(&self) -> &str {
        &self.cfg.host
    }

    fn port(&self) -> u16 {
        self.cfg.port
    }

    fn pid_file(&self) -> Option<&Path> {
        self.cfg.pid_file.as_deref()
    }

    fn log_dir(&self) -> Option<&Path> {
        self.cfg.log_dir.as_deref()
    }

    fn stderr_log_name(&self) -> &str {
        "llama-server-stderr.log"
    }

    /// `GET /health` → 200, via a short-timeout ureq agent. This is the one HTTP
    /// touchpoint — kept in the policy so `shrike-process` carries no HTTP dep.
    fn health_check(&self, base_url: &str) -> bool {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(2))
            .build();
        let url = format!("{base_url}/health");
        matches!(agent.get(&url).call(), Ok(r) if r.status() == 200)
    }

    fn process_name(&self) -> &str {
        "llama-server"
    }

    fn describe(&self) -> String {
        format!(
            "model={}, host={}, port={}",
            self.model(),
            self.cfg.host,
            self.cfg.port
        )
    }
}

/// The llama-server lifecycle manager: spawn + health-wait + orphan reaping +
/// escalating stop, all over the generic [`Supervisor`]. NOT an embedder — it
/// produces a healthy loopback endpoint the host hands to `shrike-embed-remote`.
pub struct LlamaServerManager {
    supervisor: Supervisor<LlamaPolicy>,
}

impl LlamaServerManager {
    /// Build a manager from a [`LlamaServerConfig`] (text embeddings server, no
    /// projectors). The config carries the model/host/port and the optional
    /// tuning/passthrough/pid-file.
    pub fn new(cfg: LlamaServerConfig) -> Self {
        Self {
            supervisor: Supervisor::new(LlamaPolicy {
                cfg,
                embeddings: true,
                mmprojs: Vec::new(),
            }),
        }
    }

    /// Reconfigure as a *chat* server (no `--embeddings`) with an optional
    /// multimodal projector — the shape a remote-describe deployment runs (#433):
    /// `llama-server -m model.gguf --mmproj proj.gguf`. A builder on the manager
    /// (not a config field) so the existing config constructors stay valid.
    /// Vision models want a generous `context_size` — image tokens are
    /// expensive.
    pub fn chat_mode(mut self, mmproj: Option<String>) -> Self {
        // A chat/embeddings reshape must happen before spawn; rebuild the
        // (not-yet-started) supervisor around the reshaped policy.
        let LlamaPolicy { cfg, .. } = self.take_policy();
        self.supervisor = Supervisor::new(LlamaPolicy {
            cfg,
            embeddings: false,
            mmprojs: mmproj.into_iter().collect(),
        });
        self
    }

    /// Load multimodal projector(s) on an *embeddings* server (#501): the shape
    /// behind a multimodal `embedders:` entry served by the managed
    /// llama-server. Repeatable because per-modality mmprojs load together
    /// (vision + audio for an omni model). Keeps `--embeddings` on.
    pub fn with_mmprojs(mut self, mmprojs: Vec<String>) -> Self {
        let LlamaPolicy {
            cfg, embeddings, ..
        } = self.take_policy();
        self.supervisor = Supervisor::new(LlamaPolicy {
            cfg,
            embeddings,
            mmprojs,
        });
        self
    }

    /// Move the policy out of the (pre-spawn) supervisor so a builder can reshape
    /// it, leaving a throwaway placeholder behind. Sound pre-spawn only: no child
    /// exists yet, and the chat/mmproj builders are only ever chained on a fresh
    /// `new`.
    fn take_policy(&mut self) -> LlamaPolicy {
        let placeholder = Supervisor::new(LlamaPolicy {
            cfg: LlamaServerConfig::default(),
            embeddings: true,
            mmprojs: Vec::new(),
        });
        std::mem::replace(&mut self.supervisor, placeholder).into_policy()
    }

    /// The shared observed-PID cell: `Some` while a child is believed alive.
    /// Hosts read this instead of locking the manager when the lifecycle lock may
    /// be held.
    pub fn pid_cell(&self) -> std::sync::Arc<std::sync::Mutex<Option<u32>>> {
        self.supervisor.pid_cell()
    }

    /// `http://{host}:{port}` — the endpoint the server serves.
    pub fn url(&self) -> &str {
        self.supervisor.url()
    }

    /// The live child's PID, or `None`.
    pub fn pid(&self) -> Option<u32> {
        self.supervisor.pid()
    }

    /// True while the spawned child is alive (a poll, not a cached flag).
    pub fn running(&mut self) -> bool {
        self.supervisor.running()
    }

    /// Spawn llama-server and wait for it to become healthy. Reaps any orphan
    /// first. On health timeout the child is stopped and the error carries its
    /// exit code (if it died).
    pub fn start(&mut self) -> NativeResult<()> {
        self.supervisor.start()
    }

    /// Stop the child: SIGTERM, wait up to [`SHUTDOWN_TIMEOUT`], then SIGKILL.
    /// Clears the PID file.
    pub fn stop(&mut self) {
        self.supervisor.stop()
    }

    /// The effective passthrough (reserved flags stripped). `warn` logs each
    /// rejection; the fingerprint path passes `false`. Exposed for the binding's
    /// fingerprint `args=` suffix.
    pub fn passthrough_tokens(&self, warn: bool) -> Vec<String> {
        self.supervisor.policy().passthrough_tokens(warn)
    }

    /// The exact argv this manager would spawn — for tests and diagnostics.
    pub fn build_command(&self, binary: &str) -> Vec<String> {
        self.supervisor.policy().build_command(binary)
    }

    /// Resolve the binary (override > `LLAMA_SERVER_PATH` > PATH), validated.
    pub fn find_binary(&self) -> NativeResult<String> {
        self.supervisor.policy().find_binary()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Env mutation is process-global — serialize the tests that touch it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn manager(extra: &[&str]) -> LlamaServerManager {
        manager_with_model(ModelSpec::Single("/models/test.gguf".into()), extra)
    }

    fn manager_with_model(model: ModelSpec, extra: &[&str]) -> LlamaServerManager {
        LlamaServerManager::new(
            LlamaServerConfig::new(model, "127.0.0.1", 18373)
                .with_extra_args(extra.iter().map(|s| s.to_string()).collect()),
        )
    }

    #[test]
    fn command_construction_is_exact_and_passthrough_last() {
        let cfg = LlamaServerConfig::new(
            ModelSpec::Single("/models/test.gguf".into()),
            "127.0.0.1",
            18373,
        )
        .with_extra_args(vec!["--flash-attn --ubatch-size 256".into()])
        .with_context_size(Some(512))
        .with_pooling(Some("last".into()));
        let m = LlamaServerManager::new(cfg);
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
    fn router_mode_emits_models_dir_and_max_not_model() {
        // #566: router mode loads a directory of models (request-`model`-field
        // routing), so `--models-dir`/`--models-max` replace `--model`. Global
        // defaults (--embeddings, --ctx-size, passthrough) still apply.
        let cfg = LlamaServerConfig::new(
            ModelSpec::Router {
                dir: "/models/router".into(),
                max: Some(3),
            },
            "127.0.0.1",
            18373,
        )
        .with_extra_args(vec!["--flash-attn".into()])
        .with_context_size(Some(2048));
        let m = LlamaServerManager::new(cfg);
        let cmd = m.build_command("/bin/llama-server");
        assert_eq!(
            cmd,
            vec![
                "/bin/llama-server",
                "--models-dir",
                "/models/router",
                "--models-max",
                "3",
                "--host",
                "127.0.0.1",
                "--port",
                "18373",
                "--embeddings",
                "--ctx-size",
                "2048",
                "--flash-attn",
            ]
        );
        // Never a `--model` in router mode (the two are mutually exclusive).
        assert!(!cmd.contains(&"--model".to_string()), "{cmd:?}");
        assert!(!cmd.contains(&"-m".to_string()), "{cmd:?}");
    }

    #[test]
    fn router_mode_omits_models_max_when_unset() {
        // `max: None` → let the server pick its default; emit just --models-dir.
        let m = manager_with_model(
            ModelSpec::Router {
                dir: "/models/router".into(),
                max: None,
            },
            &[],
        );
        let cmd = m.build_command("/bin/llama-server");
        assert!(cmd.contains(&"--models-dir".to_string()), "{cmd:?}");
        assert!(!cmd.contains(&"--models-max".to_string()), "{cmd:?}");
        let i = cmd.iter().position(|t| t == "--models-dir").unwrap();
        assert_eq!(cmd[i + 1], "/models/router");
    }

    #[test]
    fn router_mode_omits_model_specific_mmprojs() {
        // mmprojs are per-model; a router serves many, so no single projector
        // applies globally — they're suppressed in router mode (a passthrough
        // --mmproj is reserved-stripped regardless, tested elsewhere).
        let m = manager_with_model(
            ModelSpec::Router {
                dir: "/models/router".into(),
                max: None,
            },
            &[],
        )
        .with_mmprojs(vec!["/models/vision.mmproj.gguf".into()]);
        let cmd = m.build_command("/bin/llama-server");
        assert!(!cmd.contains(&"--mmproj".to_string()), "{cmd:?}");
    }

    #[test]
    fn router_flags_are_reserved_from_passthrough() {
        // The single-vs-router shape is `ModelSpec`-owned: a passthrough can't
        // smuggle --models-dir/--models-max (or their values) past the guard.
        let m = manager(&["--models-dir /evil --models-max 99 --flash-attn"]);
        assert_eq!(m.passthrough_tokens(false), vec!["--flash-attn"]);
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
        // EMBEDDINGS server — `--embeddings` stays on, one `--mmproj` each, in
        // declaration order.
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
        // The typed chat_mode owns the flag — a passthrough --mmproj (and its
        // value) is stripped like any reserved flag.
        let m = manager(&["--mmproj /evil/proj.gguf --flash-attn"]);
        assert_eq!(m.passthrough_tokens(false), vec!["--flash-attn"]);
    }

    #[test]
    fn reserved_flags_are_stripped_with_their_values() {
        // `--host 0.0.0.0` loses flag AND value; `--port=9` is one token; a
        // trailing reserved flag with no value doesn't panic.
        let m = manager(&["--host 0.0.0.0 --flash-attn", "--port=9999", "-m"]);
        assert_eq!(m.passthrough_tokens(false), vec!["--flash-attn"]);
    }

    #[test]
    fn passthrough_preserves_a_non_reserved_value_pair_and_order() {
        // The Peekable state machine must NOT over-consume: a non-reserved
        // value-taking flag keeps both its tokens, reserved pairs interleaved
        // between them are still stripped with their values, and the surviving
        // order is preserved.
        let m =
            manager(&["--flash-attn --host 0.0.0.0 --ubatch-size 256 -m /x --tensor-split 1,2"]);
        assert_eq!(
            m.passthrough_tokens(false),
            vec![
                "--flash-attn",
                "--ubatch-size",
                "256",
                "--tensor-split",
                "1,2"
            ]
        );
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
        let cfg = LlamaServerConfig::new(
            ModelSpec::Single("/models/test.gguf".into()),
            "127.0.0.1",
            18373,
        )
        .with_binary(Some(bin.to_string_lossy().into_owned()));
        let m = LlamaServerManager::new(cfg);
        assert_eq!(m.find_binary().unwrap(), bin.to_string_lossy());

        // A bad env path errors loud, never falls through to PATH.
        std::env::set_var("LLAMA_SERVER_PATH", "/nonexistent/binary");
        let m = manager(&[]);
        let err = m.find_binary().expect_err("must fail");
        assert!(err.to_string().contains("does not point to an executable"));
        std::env::remove_var("LLAMA_SERVER_PATH");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
