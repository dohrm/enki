//! `search` / `answer` tool schemas and their argument shapes.

use genai::chat::Tool;
use serde::Deserialize;
use serde_json::json;

pub(super) fn search_tool() -> Tool {
    Tool::new("search")
        .with_description(
            "Search the knowledge library with a natural-language query. \
             Returns passages, each with a short handle (e.g. \"s3\") to cite.",
        )
        .with_schema(json!({
            "type": "object",
            "properties": { "query": { "type": "string", "description": "The search query." } },
            "required": ["query"]
        }))
}

pub(super) fn answer_tool() -> Tool {
    Tool::new("answer")
        .with_description("Provide the final grounded answer with citations.")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "answer_markdown": { "type": "string", "description": "Final answer, with [handle] markers inline." },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "passage": { "type": "string", "description": "A passage handle, e.g. \"s3\"." },
                            "quote": { "type": "string", "description": "Verbatim text from that passage." }
                        },
                        "required": ["passage", "quote"]
                    }
                },
                "coverage": {
                    "type": "object",
                    "properties": {
                        "answered": { "type": "boolean" },
                        "gaps": { "type": "string" }
                    },
                    "required": ["answered"]
                }
            },
            "required": ["answer_markdown", "citations", "coverage"]
        }))
}

#[derive(Deserialize)]
pub(super) struct SearchArgs {
    pub(super) query: String,
}

#[derive(Deserialize, Default)]
pub(super) struct AnswerArgs {
    #[serde(default)]
    pub(super) answer_markdown: String,
    #[serde(default)]
    pub(super) citations: Vec<CitationArg>,
    #[serde(default)]
    pub(super) coverage: Option<CoverageArg>,
}

#[derive(Deserialize)]
pub(super) struct CitationArg {
    pub(super) passage: String,
    #[serde(default)]
    pub(super) quote: String,
}

#[derive(Deserialize)]
pub(super) struct CoverageArg {
    #[serde(default)]
    pub(super) answered: bool,
    #[serde(default)]
    pub(super) gaps: Option<String>,
}

/// Truncate to `n` characters, appending an ellipsis when cut.
pub(super) fn snippet(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{t}…")
    } else {
        t
    }
}
