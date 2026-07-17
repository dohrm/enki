//! Integration eval harness: replay a fixed question set against a live provider
//! and measure **relevance** (did the answer contain the expected facts / cite the
//! expected source?) and **latency**. Re-run it verbatim on a different provider —
//! swap `ENKI_LLM_MODEL` / `ENKI_LLM_ENDPOINT` — to compare, since the embedder
//! (hence the index) stays fixed.
//!
//! Prep once (embedder-side; independent of the LLM provider under test):
//!
//!   export ENKI_MANIFEST=examples/ressources/dnd/books/FR_SRD_CC_v5_2_1/document.manifest.json
//!   export ENKI_CACHE_DIR=.cache-dnd
//!   export ENKI_LIBRARY_SCOPE="D&D 5e SRD rulebook and campaign notes"
//!   cargo run -- index                                              # build corpus
//!   cargo run --example ingest_campagne -- examples/ressources/dnd/campagne   # build instance
//!
//! Then run (repeat per provider):
//!
//!   cargo run --example eval
//!   cargo run --example eval -- examples/ressources/dnd/eval_questions.json    # custom set

use anyhow::{Context, Result};
use enki::config::Config;
use enki::library::Library;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct EvalCase {
    id: String,
    category: String,
    question: String,
    /// Every substring must appear in the answer (accent/case-insensitive).
    #[serde(default)]
    expect_all: Vec<String>,
    /// At least one substring must appear (empty = no constraint).
    #[serde(default)]
    expect_any: Vec<String>,
    /// At least one cited source/label must contain one of these (empty = skip).
    #[serde(default)]
    expect_source: Vec<String>,
    /// Expected `coverage.answered` (default true; set false for out-of-corpus).
    #[serde(default = "default_true")]
    expect_answered: bool,
    #[serde(default)]
    #[allow(dead_code)]
    note: String,
}

fn default_true() -> bool {
    true
}

/// Lowercase + strip common French accents, for robust substring matching.
fn fold(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            other => other,
        })
        .collect()
}

fn contains(hay: &str, needle: &str) -> bool {
    fold(hay).contains(&fold(needle))
}

struct Outcome {
    id: String,
    category: String,
    latency_ms: u128,
    content_ok: bool,
    source_ok: bool,
    answered_ok: bool,
}

impl Outcome {
    fn pass(&self) -> bool {
        self.content_ok && self.source_ok && self.answered_ok
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    enki::telemetry::init_console();
    let cfg = Config::from_env();

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/ressources/dnd/eval_questions.json".to_string());
    let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?;
    let cases: Vec<EvalCase> = serde_json::from_str(&raw).context("parsing question set")?;

    let library = Library::open(&cfg).context(
        "opening library — did you build the index and ingest the campaign? (see the header)",
    )?;

    println!("# Enki eval");
    println!("- provider : {} @ {}", cfg.llm.model, cfg.llm.endpoint);
    println!("- embedder : {} @ {}", cfg.embed.model, cfg.embed.endpoint);
    println!("- collections : {}", library.collection_count());
    println!("- questions : {}\n", cases.len());

    // Warm-up: first call pays the model cold-load on Ollama; keep it out of stats.
    print!("warming up… ");
    let _ = library.ask("Bonjour").await;
    println!("done\n");

    println!("| id | cat | ms | content | source | answered | pass |");
    println!("|----|-----|----|---------|--------|----------|------|");

    let mut outcomes = Vec::new();
    for case in &cases {
        let t0 = Instant::now();
        let answer = library.ask(&case.question).await;
        let latency_ms = t0.elapsed().as_millis();

        let outcome = match answer {
            Ok(ans) => {
                let md = &ans.markdown;
                let content_ok = case.expect_all.iter().all(|s| contains(md, s))
                    && (case.expect_any.is_empty()
                        || case.expect_any.iter().any(|s| contains(md, s)));
                let source_ok = case.expect_source.is_empty()
                    || ans.citations.iter().any(|c| {
                        c.provenance.as_ref().is_some_and(|p| {
                            case.expect_source
                                .iter()
                                .any(|s| contains(&p.source, s) || contains(&p.label, s))
                        })
                    });
                let answered_ok = ans.coverage.answered == case.expect_answered;
                Outcome {
                    id: case.id.clone(),
                    category: case.category.clone(),
                    latency_ms,
                    content_ok,
                    source_ok,
                    answered_ok,
                }
            }
            Err(e) => {
                eprintln!("  ! {} errored: {e}", case.id);
                Outcome {
                    id: case.id.clone(),
                    category: case.category.clone(),
                    latency_ms,
                    content_ok: false,
                    source_ok: false,
                    answered_ok: false,
                }
            }
        };

        let mark = |b: bool| if b { "✓" } else { "✗" };
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            outcome.id,
            outcome.category,
            outcome.latency_ms,
            mark(outcome.content_ok),
            mark(outcome.source_ok),
            mark(outcome.answered_ok),
            mark(outcome.pass()),
        );
        outcomes.push(outcome);
    }

    report(&outcomes);
    Ok(())
}

fn report(outcomes: &[Outcome]) {
    let total = outcomes.len();
    let passed = outcomes.iter().filter(|o| o.pass()).count();

    let mut lat: Vec<u128> = outcomes.iter().map(|o| o.latency_ms).collect();
    lat.sort_unstable();
    let sum: u128 = lat.iter().sum();
    let mean = if total > 0 { sum / total as u128 } else { 0 };
    let median = lat.get(total / 2).copied().unwrap_or(0);
    let p95 = lat.get((total * 95) / 100).copied().unwrap_or(0);

    println!("\n## Summary");
    println!(
        "- pass : {passed}/{total} ({:.0}%)",
        100.0 * passed as f64 / total.max(1) as f64
    );
    println!("- latency : mean {mean} ms · median {median} ms · p95 {p95} ms");
    println!("- wall (excl. warm-up) : {} ms", sum);

    // Per-category breakdown.
    let mut cats: Vec<&str> = outcomes.iter().map(|o| o.category.as_str()).collect();
    cats.sort_unstable();
    cats.dedup();
    println!("\n## By category");
    for cat in cats {
        let group: Vec<&Outcome> = outcomes.iter().filter(|o| o.category == cat).collect();
        let ok = group.iter().filter(|o| o.pass()).count();
        println!("- {cat} : {ok}/{}", group.len());
    }

    // Machine-readable line for cross-provider diffing.
    println!(
        "\nRESULT_JSON {{\"pass\":{passed},\"total\":{total},\"mean_ms\":{mean},\"median_ms\":{median},\"p95_ms\":{p95}}}"
    );
}
