//! Pure retrieval eval — **no LLM**. Calls `Library::search` directly and checks
//! whether the passages an answer needs are in the top-k. This isolates retrieval
//! quality from generation noise (the reason the end-to-end eval couldn't tell
//! dense from BM25). Deterministic and fast (~200ms/question).
//!
//! Gold = the same `expect_*` fields as the end-to-end set, matched against the
//! retrieved passages (label + source + text) instead of the LLM answer.
//!
//!   ENKI_LEXICAL=none    cargo run --features tantivy --example retrieval_eval   # dense
//!   ENKI_LEXICAL=tantivy cargo run --features tantivy --example retrieval_eval   # hybrid
//!
//! Reports recall@5 and recall@20 (gold in top-20 but not top-5 = a reranker's
//! opportunity) plus MRR.

use anyhow::{Context, Result};
use enki::config::Config;
use enki::library::Library;
use serde::Deserialize;

const K: usize = 20;
const CUT: usize = 5; // production top-k

#[derive(Deserialize)]
struct EvalCase {
    id: String,
    category: String,
    question: String,
    #[serde(default)]
    expect_all: Vec<String>,
    #[serde(default)]
    expect_any: Vec<String>,
    #[serde(default)]
    expect_source: Vec<String>,
    #[serde(default = "yes")]
    expect_answered: bool,
}
fn yes() -> bool {
    true
}

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

/// A retrieved passage, pre-folded for matching.
struct Passage {
    origin: String, // label + source (where source-match is checked)
    full: String,   // origin + text (where term-presence is checked)
}

/// Is a passage "relevant" to the question — matches a source hint, or contains
/// any expected term? Used to locate the first relevant rank (MRR).
fn relevant(p: &Passage, case: &EvalCase) -> bool {
    if !case.expect_source.is_empty() {
        return case
            .expect_source
            .iter()
            .any(|s| p.origin.contains(&fold(s)));
    }
    case.expect_all
        .iter()
        .chain(&case.expect_any)
        .any(|t| p.full.contains(&fold(t)))
}

/// Does the passage prefix collectively satisfy every expectation?
fn covered(prefix: &[Passage], case: &EvalCase) -> bool {
    let all_ok = case
        .expect_all
        .iter()
        .all(|t| prefix.iter().any(|p| p.full.contains(&fold(t))));
    let any_ok = case.expect_any.is_empty()
        || case
            .expect_any
            .iter()
            .any(|t| prefix.iter().any(|p| p.full.contains(&fold(t))));
    let source_ok = case.expect_source.is_empty()
        || prefix.iter().any(|p| {
            case.expect_source
                .iter()
                .any(|s| p.origin.contains(&fold(s)))
        });
    all_ok && any_ok && source_ok
}

fn missed_terms(prefix: &[Passage], case: &EvalCase) -> Vec<String> {
    case.expect_all
        .iter()
        .filter(|t| !prefix.iter().any(|p| p.full.contains(&fold(t))))
        .cloned()
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    enki::telemetry::init_console();
    let cfg = Config::from_env();
    let lexical = cfg.retrieval.lexical.clone();

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/ressources/dnd/eval_questions.json".to_string());
    let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?;
    let cases: Vec<EvalCase> = serde_json::from_str(&raw).context("parsing question set")?;

    let library = Library::open(&cfg)
        .await
        .context("opening library (build index + ingest first)")?;

    println!("# Enki retrieval eval (no LLM)");
    println!("- lexical : {lexical}");
    println!("- k : {K} (cut @{CUT})\n");
    println!("| id | cat | @{CUT} | @{K} | rank | missed@{K} |");
    println!("|----|-----|----|----|------|--------|");

    let (mut hit5, mut hit20, mut mrr_sum, mut n) = (0usize, 0usize, 0f64, 0usize);
    let mut per_cat: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

    for case in &cases {
        // Negatives have no gold passage — retrieval recall is undefined for them.
        if !case.expect_answered {
            continue;
        }
        n += 1;

        let hits = library.search(&case.question, K).await?;
        let passages: Vec<Passage> = hits
            .iter()
            .map(|s| {
                let origin = fold(&format!(
                    "{} {}",
                    s.chunk.provenance.label, s.chunk.provenance.source
                ));
                let full = format!("{} {}", origin, fold(&s.chunk.text));
                Passage { origin, full }
            })
            .collect();

        let top5 = &passages[..passages.len().min(CUT)];
        let c5 = covered(top5, case);
        let c20 = covered(&passages, case);
        let rank = passages
            .iter()
            .position(|p| relevant(p, case))
            .map(|i| i + 1);

        if c5 {
            hit5 += 1;
        }
        if c20 {
            hit20 += 1;
        }
        mrr_sum += rank.map(|r| 1.0 / r as f64).unwrap_or(0.0);
        let entry = per_cat.entry(case.category.clone()).or_default();
        entry.1 += 1;
        if c5 {
            entry.0 += 1;
        }

        let mark = |b: bool| if b { "✓" } else { "✗" };
        let missed = missed_terms(&passages, case).join(", ");
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            case.id,
            case.category,
            mark(c5),
            mark(c20),
            rank.map(|r| r.to_string()).unwrap_or_else(|| "—".into()),
            missed,
        );
    }

    println!("\n## Summary");
    println!(
        "- recall@{CUT} : {hit5}/{n} ({:.0}%)",
        100.0 * hit5 as f64 / n.max(1) as f64
    );
    println!(
        "- recall@{K} : {hit20}/{n} ({:.0}%)",
        100.0 * hit20 as f64 / n.max(1) as f64
    );
    println!("- MRR : {:.3}", mrr_sum / n.max(1) as f64);
    println!("\n## recall@{CUT} by category");
    for (cat, (ok, tot)) in &per_cat {
        println!("- {cat} : {ok}/{tot}");
    }
    println!(
        "\nRESULT_JSON {{\"lexical\":\"{lexical}\",\"recall_at_{CUT}\":{hit5},\"recall_at_{K}\":{hit20},\"n\":{n},\"mrr\":{:.3}}}",
        mrr_sum / n.max(1) as f64
    );
    Ok(())
}
