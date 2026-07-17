//! CLI demo for the `enki` library.
//!
//!   cargo run -- index            # build the corpus index (occasional)
//!   cargo run -- ask "..."        # blocking, prints the resolved answer
//!   cargo run -- stream "..."     # streams the answer as it is generated

use anyhow::Result;
use enki::agent::{AgentEvent, Answer};
use enki::config::Config;
use enki::embed::{Embedder, GenaiEmbedder};
use enki::library::Library;
use enki::store::{Index, LocalStore};
use enki::{indexing, providers};
use futures::StreamExt;
use std::io::Write;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    enki::telemetry::init_console();
    let cfg = Config::from_env();
    let mut args = std::env::args().skip(1);

    match args.next().as_deref() {
        Some("index") => run_index(&cfg).await,
        Some("stream") => run_stream(&cfg, &args.collect::<Vec<_>>().join(" ")).await,
        Some("ask") => run_ask(&cfg, &args.collect::<Vec<_>>().join(" ")).await,
        Some(first) => {
            let rest: Vec<String> = args.collect();
            let question = std::iter::once(first.to_string())
                .chain(rest)
                .collect::<Vec<_>>()
                .join(" ");
            run_ask(&cfg, &question).await
        }
        None => run_ask(&cfg, "").await,
    }
}

fn question_or_default(q: &str) -> &str {
    if q.trim().is_empty() {
        "What is this library about?"
    } else {
        q
    }
}

/// Indexing pipeline: manifest → documents → store (chunked, embedded, persisted).
async fn run_index(cfg: &Config) -> Result<()> {
    let docs = indexing::manifest_documents(&cfg.indexing.manifest_path)?;
    println!("documents: {}", docs.len());

    let embedder: Arc<dyn Embedder> = Arc::new(GenaiEmbedder::new(
        providers::client(
            &cfg.embed.provider,
            &cfg.embed.endpoint,
            cfg.embed.api_key.as_deref(),
        ),
        cfg.embed.model.clone(),
    ));
    let mut store = LocalStore::open(&cfg.retrieval.cache_dir, "corpus", embedder);
    store.upsert(docs).await?;
    println!("index ready (collection `corpus`)");
    Ok(())
}

async fn run_ask(cfg: &Config, question: &str) -> Result<()> {
    let question = question_or_default(question);
    let library = Library::open(cfg)?;
    println!(
        "\n> {question}  ({} collection(s))\n",
        library.collection_count()
    );
    let answer = library.ask(question).await?;
    println!("{}\n", answer.markdown);
    print_sources(&answer);
    print_checks(&answer);
    Ok(())
}

async fn run_stream(cfg: &Config, question: &str) -> Result<()> {
    let question = question_or_default(question);
    let library = Library::open(cfg)?;
    println!(
        "\n> {question}  ({} collection(s))\n",
        library.collection_count()
    );

    let mut stream = std::pin::pin!(library.stream(question));
    let mut answer = None;
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Search(q) => eprintln!("  search: {q}"),
            AgentEvent::Token(t) => {
                print!("{t}");
                std::io::stdout().flush().ok();
            }
            AgentEvent::Done(a) => answer = Some(a),
        }
    }
    println!();
    if let Some(a) = answer {
        print_sources(&a);
        print_checks(&a);
    }
    Ok(())
}

fn print_sources(a: &Answer) {
    if a.citations.is_empty() {
        println!("(no citations)");
        return;
    }
    println!("Sources:");
    for c in &a.citations {
        match &c.provenance {
            Some(p) => {
                let loc = p
                    .page_range
                    .map(|(x, y)| format!(" (pp.{x}-{y})"))
                    .unwrap_or_default();
                println!("  [{}] {} — {}{}", c.number, p.source, p.label, loc);
            }
            None => println!("  [{}] hallucinated handle: {}", c.number, c.handle),
        }
    }
}

fn print_checks(a: &Answer) {
    let quoted: Vec<_> = a.citations.iter().filter(|c| c.quote.is_some()).collect();
    if !quoted.is_empty() {
        println!("\nCitation check:");
        for c in quoted {
            println!(
                "  {} [{}]",
                if c.quote_verified { "ok" } else { "NOT FOUND" },
                c.number
            );
        }
    }
    if !a.coverage.answered {
        println!(
            "\npartial coverage: {}",
            a.coverage.gaps.clone().unwrap_or_default()
        );
    }
}
