//! Evidence gathering: run `search` rounds until the model stops, the budget is
//! reached, or it starts spinning on stale/duplicate queries.

use super::chat_options;
use super::tools::{SearchArgs, snippet};
use crate::model::Scored;
use crate::registry::Registry;
use crate::retrieval::Filters;
use crate::search::SearchEngine;
use anyhow::{Context, Result};
use genai::Client;
use genai::chat::{ChatMessage, ChatRequest, ToolResponse};
use serde_json::json;
use std::collections::HashSet;

/// Consecutive stale searches tolerated before `gather` forces the answer.
const STALE_SEARCH_LIMIT: usize = 2;

/// Runs search rounds until the model stops requesting searches or the budget is
/// reached. `on_search` is called with each query. Returns the accumulated chat
/// (tool results appended, chunks interned into `registry`).
#[allow(clippy::too_many_arguments)]
pub(super) async fn gather(
    client: &Client,
    model: &str,
    mut chat: ChatRequest,
    engine: &SearchEngine,
    registry: &mut Registry,
    top_k: usize,
    max_rounds: usize,
    filters: &Filters,
    mut on_search: impl FnMut(&str),
) -> Result<ChatRequest> {
    // Some models spin: they re-query the same entity with tweaked wording, never
    // converging. A search is "stale" if it repeats an earlier query (normalised)
    // or interns no new chunk; after a couple in a row we stop gathering and force
    // the answer with what we already have — this also keeps the context from
    // bloating, which some models answer poorly from.
    let mut seen_queries: HashSet<String> = HashSet::new();
    let mut stale = 0usize;
    let mut stopped = false;
    for round in 0..max_rounds {
        let t = std::time::Instant::now();
        let resp = client
            .exec_chat(model, chat.clone(), Some(&chat_options()))
            .await
            .context("exec_chat (gather)")?;
        let calls: Vec<_> = resp
            .into_tool_calls()
            .into_iter()
            .filter(|tc| tc.fn_name == "search")
            .collect();
        tracing::debug!(
            target: "enki",
            round,
            searches = calls.len(),
            elapsed_ms = t.elapsed().as_millis(),
            "llm round"
        );
        if calls.is_empty() {
            stopped = true;
            break; // model is done gathering
        }
        chat = chat.append_message(ChatMessage::from(calls.clone()));
        for tc in calls {
            let args: SearchArgs =
                serde_json::from_value(tc.fn_arguments.clone()).context("parse search args")?;
            on_search(&args.query);
            let duplicate = !seen_queries.insert(normalize_query(&args.query));
            let before = registry.len();
            let payload = do_search(&args.query, engine, registry, top_k, filters).await?;
            if duplicate || registry.len() == before {
                stale += 1;
            } else {
                stale = 0;
            }
            chat = chat.append_message(ChatMessage::from(ToolResponse::new(tc.call_id, payload)));
        }
        if stale >= STALE_SEARCH_LIMIT {
            tracing::debug!(target: "enki", round, stale, "gather: stale searches, forcing answer");
            stopped = true;
            break;
        }
    }
    if !stopped {
        tracing::warn!(target: "enki", max_rounds, "gather: round budget exhausted, forcing answer");
    }
    Ok(chat)
}

async fn do_search(
    query: &str,
    engine: &SearchEngine,
    registry: &mut Registry,
    top_k: usize,
    filters: &Filters,
) -> Result<String> {
    let hits = engine.search(query, top_k, filters.clone()).await?;
    let mut passages = Vec::new();
    for Scored { chunk, score } in &hits {
        let handle = registry.intern(chunk);
        passages.push(json!({
            "handle": handle,
            "section": chunk.provenance.label,
            "pages": chunk.provenance.page_range.map(|(a, b)| format!("{a}-{b}")),
            "score": score,
            "text": snippet(&chunk.text, 1200),
        }));
    }
    Ok(serde_json::to_string(&json!({ "passages": passages }))?)
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
