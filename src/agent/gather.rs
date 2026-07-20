//! Evidence gathering: run tool rounds (`search`, plus `neighbors` / `open` when a
//! relation graph is loaded) until the model stops, the budget is reached, or it
//! starts spinning on stale/duplicate calls.

use super::chat_options;
use super::tools::{NeighborsArgs, OpenArgs, SearchArgs, snippet};
use crate::graph::{Direction, GraphStore};
use crate::model::{Chunk, Scored};
use crate::registry::Registry;
use crate::retrieval::Filters;
use crate::search::SearchEngine;
use anyhow::{Context, Result};
use genai::Client;
use genai::chat::{ChatMessage, ChatRequest, ToolResponse};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Consecutive stale rounds tolerated before `gather` forces the answer.
const STALE_SEARCH_LIMIT: usize = 2;
/// Default neighbour cap when the model doesn't specify one.
const DEFAULT_NEIGHBORS: usize = 10;

/// Runs tool rounds until the model stops requesting them or the budget is reached.
/// `on_search` is called with each `search` query. Returns the accumulated chat
/// (tool results appended, chunks interned into `registry`).
#[allow(clippy::too_many_arguments)]
pub(super) async fn gather(
    client: &Client,
    model: &str,
    mut chat: ChatRequest,
    engine: &SearchEngine,
    graph: Option<&Arc<dyn GraphStore>>,
    registry: &mut Registry,
    top_k: usize,
    max_rounds: usize,
    filters: &Filters,
    mut on_search: impl FnMut(&str),
) -> Result<ChatRequest> {
    // Some models spin: they re-query the same entity with tweaked wording, never
    // converging. A round is "stale" if it repeats a query (normalised) or interns
    // no new chunk; after a couple in a row we stop gathering and force the answer
    // with what we already have — this also keeps the context from bloating, which
    // some models answer poorly from.
    let mut seen_queries: HashSet<String> = HashSet::new();
    let mut stale = 0usize;
    let mut stopped = false;
    for round in 0..max_rounds {
        let t = std::time::Instant::now();
        let resp = client
            .exec_chat(model, chat.clone(), Some(&chat_options()))
            .await
            .context("exec_chat (gather)")?;
        let calls: Vec<_> = resp.into_tool_calls();
        tracing::debug!(
            target: "enki",
            round,
            calls = calls.len(),
            elapsed_ms = t.elapsed().as_millis(),
            "llm round"
        );
        if calls.is_empty() {
            stopped = true;
            break; // model is done gathering
        }
        chat = chat.append_message(ChatMessage::from(calls.clone()));
        let before = registry.len();
        let mut dup_only = true;
        for tc in calls {
            let payload = match tc.fn_name.as_str() {
                "search" => {
                    let args: SearchArgs = serde_json::from_value(tc.fn_arguments.clone())
                        .context("parse search args")?;
                    on_search(&args.query);
                    if seen_queries.insert(normalize_query(&args.query)) {
                        dup_only = false;
                    }
                    do_search(&args.query, engine, registry, top_k, filters).await?
                }
                "neighbors" => {
                    dup_only = false;
                    let args: NeighborsArgs = serde_json::from_value(tc.fn_arguments.clone())
                        .context("parse neighbors args")?;
                    do_neighbors(&args, engine, graph, registry, filters).await?
                }
                "open" => {
                    dup_only = false;
                    let args: OpenArgs = serde_json::from_value(tc.fn_arguments.clone())
                        .context("parse open args")?;
                    do_open(&args.ids, engine, registry, filters).await?
                }
                other => {
                    tracing::warn!(target: "enki", tool = other, "unknown tool call");
                    json!({ "error": format!("unknown tool `{other}`") }).to_string()
                }
            };
            chat = chat.append_message(ChatMessage::from(ToolResponse::new(tc.call_id, payload)));
        }
        // No new evidence this round (or only repeated queries) → stale.
        if registry.len() == before || dup_only {
            stale += 1;
        } else {
            stale = 0;
        }
        if stale >= STALE_SEARCH_LIMIT {
            tracing::debug!(target: "enki", round, stale, "gather: stale rounds, forcing answer");
            stopped = true;
            break;
        }
    }
    if !stopped {
        tracing::warn!(target: "enki", max_rounds, "gather: round budget exhausted, forcing answer");
    }
    Ok(chat)
}

/// Build one passage JSON object, interning the chunk. `id` lets the model pivot
/// into graph tools; `score` (search only) and `relation`/`direction` (neighbours)
/// are added when relevant.
fn passage(
    chunk: &Chunk,
    score: Option<f32>,
    relation: Option<(&str, &str)>,
    registry: &mut Registry,
) -> serde_json::Value {
    let handle = registry.intern(chunk);
    let mut v = json!({
        "handle": handle,
        "id": chunk.doc_id,
        "section": chunk.provenance.label,
        "pages": chunk.provenance.page_range.map(|(a, b)| format!("{a}-{b}")),
        "text": snippet(&chunk.text, 1200),
    });
    let obj = v.as_object_mut().expect("json object");
    if let Some(s) = score {
        obj.insert("score".into(), json!(s));
    }
    if let Some((predicate, direction)) = relation {
        obj.insert("relation".into(), json!(predicate));
        obj.insert("direction".into(), json!(direction));
    }
    v
}

async fn do_search(
    query: &str,
    engine: &SearchEngine,
    registry: &mut Registry,
    top_k: usize,
    filters: &Filters,
) -> Result<String> {
    let hits = engine.search(query, top_k, filters.clone()).await?;
    let passages: Vec<_> = hits
        .iter()
        .map(|Scored { chunk, score }| passage(chunk, Some(*score), None, registry))
        .collect();
    Ok(serde_json::to_string(&json!({ "passages": passages }))?)
}

/// Open specific entities by id (direct access, no similarity).
async fn do_open(
    ids: &[String],
    engine: &SearchEngine,
    registry: &mut Registry,
    filters: &Filters,
) -> Result<String> {
    let hits = engine.fetch(ids, filters.clone()).await?;
    let passages: Vec<_> = hits
        .iter()
        .map(|s| passage(&s.chunk, None, None, registry))
        .collect();
    Ok(serde_json::to_string(&json!({ "passages": passages }))?)
}

/// List an entity's graph neighbours and return their passages (with the relation).
async fn do_neighbors(
    args: &NeighborsArgs,
    engine: &SearchEngine,
    graph: Option<&Arc<dyn GraphStore>>,
    registry: &mut Registry,
    filters: &Filters,
) -> Result<String> {
    let Some(graph) = graph else {
        return Ok(json!({ "error": "no relation graph is loaded" }).to_string());
    };
    let limit = args.limit.unwrap_or(DEFAULT_NEIGHBORS);
    let neighbors = graph
        .neighbors(&args.id, args.predicate.as_deref(), Direction::Both, limit)
        .await?;
    if neighbors.is_empty() {
        return Ok(json!({ "neighbors": [] }).to_string());
    }

    // doc_id -> the relation that surfaced it (first wins).
    let mut rel: HashMap<&str, (&str, &str)> = HashMap::new();
    for n in &neighbors {
        rel.entry(n.id.as_str())
            .or_insert((n.predicate.as_str(), n.direction.as_str()));
    }
    let ids: Vec<String> = neighbors.iter().map(|n| n.id.clone()).collect();
    let hits = engine.fetch(&ids, filters.clone()).await?;
    let passages: Vec<_> = hits
        .iter()
        .map(|s| {
            let relation = rel.get(s.chunk.doc_id.as_str()).copied();
            passage(&s.chunk, None, relation, registry)
        })
        .collect();
    Ok(serde_json::to_string(&json!({ "neighbors": passages }))?)
}

/// Normalise a query for duplicate detection: lowercase, drop punctuation/quotes,
/// collapse whitespace. `"Lyra Murmelune"` and `Lyra  Murmelune` compare equal.
fn normalize_query(q: &str) -> String {
    q.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::normalize_query;

    #[test]
    fn dedup_ignores_quotes_case_and_spacing() {
        assert_eq!(
            normalize_query("\"Lyra Murmelune\""),
            normalize_query("lyra  murmelune")
        );
        assert_eq!(normalize_query("Boule de feu !"), "boule de feu");
    }

    #[test]
    fn distinct_queries_stay_distinct() {
        assert_ne!(
            normalize_query("Murmelune"),
            normalize_query("Lyra Murmelune")
        );
    }
}
