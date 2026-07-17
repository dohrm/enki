//! Turning model output (structured `answer` tool call, or prose) into a resolved
//! [`Answer`] with renumbered, provenance-backed citations.

use super::citations::{normalize, resolve_citations};
use super::tools::{AnswerArgs, CitationArg};
use super::{Answer, Coverage, ResolvedCitation};
use crate::registry::Registry;
use serde_json::Value;

pub(super) fn answer_from_tool(args: &Value, registry: &Registry) -> Answer {
    let a: AnswerArgs = serde_json::from_value(args.clone()).unwrap_or_default();
    let (markdown, order) = resolve_citations(&a.answer_markdown);
    let citations = build_citations(&order, &a.citations, registry);
    let coverage = a
        .coverage
        .map(|c| Coverage {
            answered: c.answered,
            gaps: c.gaps,
        })
        .unwrap_or(Coverage {
            answered: true,
            gaps: None,
        });
    Answer {
        markdown,
        citations,
        coverage,
    }
}

pub(super) fn answer_from_prose(text: &str, registry: &Registry) -> Answer {
    let (markdown, order) = resolve_citations(text);
    Answer {
        markdown,
        citations: build_citations(&order, &[], registry),
        coverage: Coverage {
            answered: true,
            gaps: None,
        },
    }
}

/// Last-resort answer when the model produced nothing usable — an honest empty
/// result rather than a blank string handed to the caller.
pub(super) fn empty_answer() -> Answer {
    Answer {
        markdown: "No answer could be produced from the retrieved passages.".to_string(),
        citations: Vec::new(),
        coverage: Coverage {
            answered: false,
            gaps: Some("the model returned an empty final response".to_string()),
        },
    }
}

/// Resolve ordered handles to citations, matching optional quotes (from the
/// structured `answer` tool) and verifying them against the passage text.
fn build_citations(
    order: &[String],
    provided: &[CitationArg],
    registry: &Registry,
) -> Vec<ResolvedCitation> {
    order
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let chunk = registry.get(h);
            let quote = provided
                .iter()
                .find(|c| c.passage.trim().to_ascii_lowercase() == *h)
                .map(|c| c.quote.clone())
                .filter(|q| !q.is_empty());
            let quote_verified = match (chunk, &quote) {
                (Some(ch), Some(q)) => normalize(&ch.text).contains(&normalize(q)),
                _ => false,
            };
            ResolvedCitation {
                number: i + 1,
                handle: h.clone(),
                provenance: chunk.map(|c| c.provenance.clone()),
                quote,
                quote_verified,
            }
        })
        .collect()
}
