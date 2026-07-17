//! Observability seam. The library instruments its hot paths with [`tracing`]
//! spans and events (search, embedding, agent rounds); it does **not** install a
//! subscriber — that is the host application's decision.
//!
//! - Want the default console sink? Call [`init_console`] once at startup (feature
//!   `console`, on by default). It writes to stderr, filtered by `ENKI_LOG`.
//! - Want your own sink (a file, JSON, an in-app buffer, a closure)? Install any
//!   `tracing` subscriber / layer instead and leave `init_console` uncalled.
//!
//! In `tracing` terms the "trait" seam is [`tracing::Subscriber`] and the "fn"
//! seam is a closure inside a layer — so enki carries no logging abstraction of
//! its own.

/// Target used by all enki spans/events, so hosts can filter with
/// `ENKI_LOG=enki=debug`.
pub const TARGET: &str = "enki";

/// Install a console (stderr) subscriber, filtered by the `ENKI_LOG` env var
/// (default `info`). Idempotent and safe to call from several entry points: if a
/// global subscriber is already set, this is a no-op.
///
/// Examples: `ENKI_LOG=debug`, `ENKI_LOG=enki=debug`, `ENKI_LOG=off`.
#[cfg(feature = "console")]
pub fn init_console() {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter = EnvFilter::try_from_env("ENKI_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    // `try_init` returns Err if a subscriber is already installed — ignore it.
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .try_init();
}

/// No-op when the `console` feature is off: the host is expected to install its
/// own `tracing` subscriber. Kept so entry points can call it unconditionally.
#[cfg(not(feature = "console"))]
pub fn init_console() {}
