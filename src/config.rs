//! Configuration, split by concern so each part of the code — and each consumer —
//! only depends on what it needs: a retrieval-only user builds an [`EmbedConfig`]
//! with a [`RetrievalConfig`] and never touches [`LlmConfig`] / [`AgentConfig`];
//! an indexing job needs no LLM config at all.
//!
//! Every group has a `from_env` reading `ENKI_*` vars; [`Config::from_env`]
//! composes them. Fields are public, so a host app can also build them directly.

use std::path::PathBuf;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_opt(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// Provider + model + optional key for one genai role. Reused for LLM and
/// embeddings, which may live on different providers.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// `ollama` (uses `endpoint`) or a genai-native provider routed by model name
    /// (`gemini`, `openai`, `anthropic`, …).
    pub provider: String,
    pub model: String,
    pub endpoint: String,
    /// Cloud API key, injected by the caller (a desktop app supplies it from its
    /// own settings — no process env var). `None` → genai reads the provider's
    /// default env var.
    pub api_key: Option<String>,
}

impl ProviderConfig {
    fn from_env(prefix: &str, default_model: &str) -> Self {
        Self {
            provider: env_or(&format!("ENKI_{prefix}_PROVIDER"), "ollama"),
            model: env_or(&format!("ENKI_{prefix}_MODEL"), default_model),
            endpoint: env_or(
                &format!("ENKI_{prefix}_ENDPOINT"),
                "http://localhost:11434/",
            ),
            api_key: env_opt(&format!("ENKI_{prefix}_API_KEY")),
        }
    }
}

/// Embeddings — needed by both indexing (write) and retrieval (query). Keep this
/// on the model an index was built with; changing it invalidates the vectors.
pub type EmbedConfig = ProviderConfig;
/// The chat LLM — needed only by the agent (inference).
pub type LlmConfig = ProviderConfig;

/// Everything to build and query the search engine — no LLM involved.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Where collections are persisted (and read from). Shared with indexing.
    pub cache_dir: PathBuf,
    /// Passages returned per search.
    pub top_k: usize,
    /// Lexical backend: `none` or `tantivy` (needs the `tantivy` feature).
    pub lexical: String,
    /// Reranker backend: `none` or `fastembed` (needs the `fastembed` feature).
    pub rerank: String,
    /// Optional cache dir for the reranker model (reuse a host app's download).
    pub rerank_cache: Option<PathBuf>,
}

impl RetrievalConfig {
    pub fn from_env() -> Self {
        Self {
            cache_dir: env_or("ENKI_CACHE_DIR", ".cache").into(),
            top_k: env_or("ENKI_TOP_K", "5").parse().unwrap_or(5),
            lexical: env_or("ENKI_LEXICAL", "none"),
            rerank: env_or("ENKI_RERANK", "none"),
            rerank_cache: env_opt("ENKI_RERANK_CACHE").map(Into::into),
        }
    }
}

/// The agentic loop: what the library is about (injected into the system prompt)
/// and how many gather rounds it may run.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Static per-library scope injected into the system prompt.
    pub scope: String,
    pub max_rounds: usize,
}

impl AgentConfig {
    pub fn from_env() -> Self {
        Self {
            scope: env_or("ENKI_LIBRARY_SCOPE", ""),
            max_rounds: env_or("ENKI_MAX_ROUNDS", "6").parse().unwrap_or(6),
        }
    }
}

/// Indexing inputs (the manifest importer). Writes into the retrieval store.
#[derive(Debug, Clone)]
pub struct IndexingConfig {
    /// Document manifest: hierarchical outline + section content.
    pub manifest_path: PathBuf,
}

impl IndexingConfig {
    pub fn from_env() -> Self {
        Self {
            manifest_path: env_or("ENKI_MANIFEST", "data/document.manifest.json").into(),
        }
    }
}

/// The whole configuration, grouped by concern. Consumers can take just the
/// groups they use (e.g. `&cfg.embed` + `&cfg.retrieval` for a retrieval harness).
#[derive(Debug, Clone)]
pub struct Config {
    pub embed: EmbedConfig,
    pub llm: LlmConfig,
    pub retrieval: RetrievalConfig,
    pub agent: AgentConfig,
    pub indexing: IndexingConfig,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            embed: EmbedConfig::from_env("EMBED", "bge-m3"),
            llm: LlmConfig::from_env("LLM", "llama3.1"),
            retrieval: RetrievalConfig::from_env(),
            agent: AgentConfig::from_env(),
            indexing: IndexingConfig::from_env(),
        }
    }
}
