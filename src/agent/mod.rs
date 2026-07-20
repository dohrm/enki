//! Homemade agentic loop over genai. Gathers evidence via the `search` tool, then
//! produces a grounded, cited answer — as a struct (`ask`) or a token stream (`stream`).
//! The library never prints; callers render the returned [`Answer`] / [`AgentEvent`]s.
//!
//! Split across: [`tools`] (schemas/args), [`gather`] (evidence loop),
//! [`answer`] (build the resolved answer), [`citations`] (handle renumbering).

mod answer;
mod citations;
mod gather;
mod tools;

use answer::{answer_from_prose, answer_from_tool, empty_answer};
use gather::gather;
use tools::{answer_tool, graph_tools, search_tool};

use crate::graph::GraphStore;
use crate::model::Provenance;
use crate::registry::Registry;
use crate::retrieval::Filters;
use crate::search::SearchEngine;
use anyhow::{Context, Result};
use futures::StreamExt;
use genai::Client;
use genai::chat::{CacheControl, ChatMessage, ChatOptions, ChatRequest, ChatStreamEvent, Tool};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Shared decoding options for every LLM call in the loop.
///
/// - Deterministic (temperature 0 + fixed seed): grounded RAG wants the same
///   answer for the same evidence, and reproducibility lets the eval measure
///   retrieval rather than sampling noise.
/// - Prompt caching (ephemeral): the system prompt + tool schemas are a stable
///   prefix shared by every gather round and the final call, so caching it cuts
///   repeated prefill. Providers that don't support it (e.g. Ollama, Gemini's
///   implicit cache) simply ignore the hint.
fn chat_options() -> ChatOptions {
    ChatOptions::default()
        .with_temperature(0.0)
        .with_seed(42)
        .with_cache_control(CacheControl::Ephemeral)
}

/// Tools offered while gathering: `search` always; graph traversal when a relation
/// graph is loaded.
fn gather_tools(has_graph: bool) -> Vec<Tool> {
    let mut tools = vec![search_tool()];
    if has_graph {
        tools.extend(graph_tools());
    }
    tools
}

const CLOSING: &str = "Answer now, grounded strictly in the passages already retrieved. \
    Cite each claim inline by its passage handle (e.g. [s3]). If the evidence is \
    insufficient, say so and set coverage.answered = false.";

// ---------- Public result types ----------

/// A grounded answer: rendered markdown (handles renumbered to `[1]`, `[2]`…),
/// resolved citations, and a coverage/honesty signal.
#[derive(Debug, Clone)]
pub struct Answer {
    pub markdown: String,
    pub citations: Vec<ResolvedCitation>,
    pub coverage: Coverage,
}

#[derive(Debug, Clone)]
pub struct ResolvedCitation {
    pub number: usize,
    pub handle: String,
    /// Resolved provenance, or `None` if the model cited a handle never retrieved.
    pub provenance: Option<Provenance>,
    pub quote: Option<String>,
    /// Whether `quote` was found (fuzzily) in the cited passage's text.
    pub quote_verified: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Coverage {
    pub answered: bool,
    pub gaps: Option<String>,
}

/// Streaming events for [`stream`].
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A retrieval query was issued while gathering evidence.
    Search(String),
    /// A chunk of the final answer text.
    Token(String),
    /// The final resolved answer (citations, coverage).
    Done(Answer),
}

// ---------- Entry points ----------

/// Blocking: gather evidence, then produce a structured [`Answer`] (uses the
/// `answer` tool for structured quotes + coverage; falls back to prose).
#[allow(clippy::too_many_arguments)]
pub async fn ask(
    client: &Client,
    model: &str,
    system: &str,
    question: &str,
    engine: &SearchEngine,
    graph: Option<&Arc<dyn GraphStore>>,
    top_k: usize,
    max_rounds: usize,
    filters: Filters,
) -> Result<Answer> {
    let started = std::time::Instant::now();
    tracing::info!(target: "enki", question, "ask");
    let mut registry = Registry::default();
    let chat = ChatRequest::from_system(system)
        .with_tools(gather_tools(graph.is_some()))
        .append_message(ChatMessage::user(question));
    let chat = gather(
        client,
        model,
        chat,
        engine,
        graph,
        &mut registry,
        top_k,
        max_rounds,
        &filters,
        |_| {},
    )
    .await?;

    let closing = chat
        .clone()
        .with_tools(vec![answer_tool()])
        .append_message(ChatMessage::user(CLOSING));
    let resp = client
        .exec_chat(model, closing, Some(&chat_options()))
        .await
        .context("exec_chat (final)")?;
    let text = resp.first_text().map(str::to_string).unwrap_or_default();
    let tool_answer = resp
        .into_tool_calls()
        .into_iter()
        .find(|tc| tc.fn_name == "answer")
        .map(|tc| answer_from_tool(&tc.fn_arguments, &registry));

    // Never hand back a blank answer: answer tool → prose text → one forced-prose
    // retry → a graceful "no answer" (some models emit an empty final turn).
    let answer = match tool_answer {
        Some(a) if !a.markdown.trim().is_empty() => a,
        _ if !text.trim().is_empty() => answer_from_prose(&text, &registry),
        _ => {
            tracing::warn!(target: "enki", "empty final answer; retrying as prose");
            let prose = chat
                .with_tools(Vec::<Tool>::new())
                .append_message(ChatMessage::user(CLOSING));
            let retry = client
                .exec_chat(model, prose, Some(&chat_options()))
                .await
                .context("exec_chat (final retry)")?;
            let retry_text = retry.first_text().unwrap_or_default().trim().to_string();
            if retry_text.is_empty() {
                empty_answer()
            } else {
                answer_from_prose(&retry_text, &registry)
            }
        }
    };

    tracing::info!(
        target: "enki",
        answered = answer.coverage.answered,
        citations = answer.citations.len(),
        elapsed_ms = started.elapsed().as_millis(),
        "answer ready"
    );
    Ok(answer)
}

/// Streaming: emits [`AgentEvent::Search`] while gathering, then streams the final
/// answer as [`AgentEvent::Token`]s, and finally [`AgentEvent::Done`] with the
/// resolved [`Answer`]. Runs the loop in a background task.
#[allow(clippy::too_many_arguments)]
pub fn stream(
    client: Client,
    model: String,
    system: String,
    question: String,
    engine: Arc<SearchEngine>,
    graph: Option<Arc<dyn GraphStore>>,
    top_k: usize,
    max_rounds: usize,
    filters: Filters,
) -> impl futures::Stream<Item = AgentEvent> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        if let Err(e) = stream_inner(
            &client,
            &model,
            &system,
            &question,
            &engine,
            graph.as_ref(),
            top_k,
            max_rounds,
            &filters,
            &tx,
        )
        .await
        {
            let _ = tx.send(AgentEvent::Token(format!("\n[error: {e}]"))).await;
        }
    });
    tokio_stream::wrappers::ReceiverStream::new(rx)
}

#[allow(clippy::too_many_arguments)]
async fn stream_inner(
    client: &Client,
    model: &str,
    system: &str,
    question: &str,
    engine: &SearchEngine,
    graph: Option<&Arc<dyn GraphStore>>,
    top_k: usize,
    max_rounds: usize,
    filters: &Filters,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<()> {
    let mut registry = Registry::default();
    let chat = ChatRequest::from_system(system)
        .with_tools(gather_tools(graph.is_some()))
        .append_message(ChatMessage::user(question));
    let chat = gather(
        client,
        model,
        chat,
        engine,
        graph,
        &mut registry,
        top_k,
        max_rounds,
        filters,
        |q| {
            let _ = tx.try_send(AgentEvent::Search(q.to_string()));
        },
    )
    .await?;

    // Final answer as streamed prose (no tools → the model must emit text).
    let closing = chat
        .with_tools(Vec::<Tool>::new())
        .append_message(ChatMessage::user(CLOSING));
    let chat_res = client
        .exec_chat_stream(model, closing, Some(&chat_options()))
        .await
        .context("exec_chat_stream")?;
    let mut stream = chat_res.stream;
    let mut full = String::new();
    while let Some(event) = stream.next().await {
        if let ChatStreamEvent::Chunk(chunk) = event? {
            full.push_str(&chunk.content);
            let _ = tx.send(AgentEvent::Token(chunk.content)).await;
        }
    }

    // Never end on a blank stream: surface the graceful message as a token too.
    let done = if full.trim().is_empty() {
        tracing::warn!(target: "enki", "empty streamed answer");
        let fallback = empty_answer();
        let _ = tx.send(AgentEvent::Token(fallback.markdown.clone())).await;
        fallback
    } else {
        answer_from_prose(&full, &registry)
    };
    let _ = tx.send(AgentEvent::Done(done)).await;
    Ok(())
}
