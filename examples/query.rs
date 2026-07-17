//! Minimal library usage: open a library and ask a question.
//! This is how a consumer (e.g. a Tauri backend) embeds enki.
//!
//!   cargo run --example query -- "your question"

use anyhow::Result;
use enki::config::Config;
use enki::library::Library;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    enki::telemetry::init_console();
    let cfg = Config::from_env();
    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is this library about?".to_string());

    // The whole wiring — store, search engine, LLM, prompt — is behind `open`.
    let library = Library::open(&cfg)?;
    let answer = library.ask(&question).await?;

    println!("{}\n", answer.markdown);
    println!("Sources:");
    for c in &answer.citations {
        match &c.provenance {
            Some(p) => println!("  [{}] {} — {}", c.number, p.source, p.label),
            None => println!("  [{}] (unresolved: {})", c.number, c.handle),
        }
    }
    if !answer.coverage.answered {
        println!("\nPartial: {}", answer.coverage.gaps.unwrap_or_default());
    }
    Ok(())
}
