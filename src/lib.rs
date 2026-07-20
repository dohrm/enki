//! **Enki** — a provider-agnostic agentic RAG library.
//!
//! Two facets: an **indexing** pipeline (manifest → sections → embeddings) and an
//! **inference** side (an agentic loop where retrieval is a *tool*, not prompt-stuffed
//! context). The LLM/provider and the domain (scope, corpus) are injected via config —
//! the library holds no opinion on either.
//!
//! See `SPEC.md` for the full design rationale.

pub mod agent;
pub mod config;
pub mod embed;
pub mod indexing;
#[cfg(feature = "tantivy")]
pub mod lexical;
pub mod library;
pub mod model;
pub mod prompt;
pub mod providers;
#[cfg(feature = "qdrant")]
pub mod qdrant;
pub mod registry;
#[cfg(feature = "fastembed")]
pub mod rerank;
pub mod retrieval;
pub mod search;
pub mod sparse;
pub mod store;
pub mod telemetry;
