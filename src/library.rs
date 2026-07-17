//! High-level facade. A consumer opens a `Library` and calls `ask` (blocking,
//! returns a structured [`Answer`]) or `stream` (token events), and `ingest` /
//! `delete` to manage living content. All the wiring (store, engine, LLM, prompt)
//! is hidden by [`Library::open`] — or bring your own via [`Library::builder`]
//! (custom embedder, custom collections, any genai client).

use crate::agent::{self, AgentEvent, Answer};
use crate::config::Config;
use crate::embed::{Embedder, GenaiEmbedder};
use crate::model::{Document, Scored};
use crate::prompt;
use crate::providers;
use crate::retrieval::{BruteForce, Filters, Retriever};
use crate::search::{Collection, Reranker, SearchEngine};
use crate::store::{Index, LocalStore, VectorStore};
use anyhow::Result;
use futures::Stream;
use genai::Client;
use std::path::PathBuf;
use std::sync::Arc;

pub struct Library {
    llm: Client,
    model: String,
    system: String,
    engine: Arc<SearchEngine>,
    cache_dir: PathBuf,
    /// Shared by the query path (engine) and the write path (`ingest` / `delete`).
    embedder: Arc<dyn Embedder>,
    top_k: usize,
    max_rounds: usize,
}

impl Library {
    /// Open a library from config: load the persisted collections (`corpus`
    /// required; `spells` / `instance` added when present), wire the search engine
    /// and LLM. Uses the genai embedder/LLM described by `cfg`.
    pub fn open(cfg: &Config) -> Result<Self> {
        let embedder: Arc<dyn Embedder> = Arc::new(GenaiEmbedder::new(
            providers::client(
                &cfg.embed.provider,
                &cfg.embed.endpoint,
                cfg.embed.api_key.as_deref(),
            ),
            cfg.embed.model.clone(),
        ));

        // Build a collection: dense (brute-force) + optional lexical (BM25), fused
        // by the engine's modality RRF. `corpus` is required; the others are
        // loaded when their store exists.
        let build = |store: VectorStore, name: &str, tier| -> Result<Collection> {
            let mut retrievers: Vec<Arc<dyn Retriever>> = vec![Arc::new(BruteForce::new(store))];
            if let Some(lex) = lexical_retriever(cfg, name)? {
                retrievers.push(lex);
            }
            Ok(Collection {
                name: name.to_string(),
                tier,
                retrievers,
            })
        };

        let mut collections = vec![build(
            VectorStore::open(&cfg.retrieval.cache_dir, "corpus")?,
            "corpus",
            0,
        )?];
        // Optional collections: `spells` (finer book chunks, tier 0), `instance`
        // (living campaign content, tier 1).
        for (name, tier) in [("spells", 0), ("instance", 1)] {
            if let Ok(store) = VectorStore::open(&cfg.retrieval.cache_dir, name) {
                collections.push(build(store, name, tier)?);
            }
        }

        let mut engine = SearchEngine::new(embedder.clone(), collections);
        if let Some(reranker) = reranker(cfg)? {
            engine = engine.with_reranker(reranker);
        }

        Ok(Self {
            llm: providers::client(
                &cfg.llm.provider,
                &cfg.llm.endpoint,
                cfg.llm.api_key.as_deref(),
            ),
            model: cfg.llm.model.clone(),
            system: prompt::system_prompt(&cfg.agent.scope, cfg.agent.max_rounds),
            engine: Arc::new(engine),
            cache_dir: cfg.retrieval.cache_dir.clone(),
            embedder,
            top_k: cfg.retrieval.top_k,
            max_rounds: cfg.agent.max_rounds,
        })
    }

    /// Assemble a library from your own parts — a custom [`Embedder`] (e.g.
    /// fastembed), a hand-built [`SearchEngine`] (custom collections / tiers /
    /// retrievers), and any genai [`Client`]. The escape hatch from the
    /// opinionated [`Library::open`], keeping `ask` / `stream` / `ingest`.
    pub fn builder(
        engine: Arc<SearchEngine>,
        embedder: Arc<dyn Embedder>,
        llm: Client,
        model: impl Into<String>,
    ) -> LibraryBuilder {
        LibraryBuilder {
            engine,
            embedder,
            llm,
            model: model.into(),
            system: String::new(),
            cache_dir: PathBuf::from(".cache"),
            top_k: 5,
            max_rounds: 6,
        }
    }

    /// Number of loaded collections.
    pub fn collection_count(&self) -> usize {
        self.engine.collection_count()
    }

    /// Raw retrieval, no agent/LLM: the fused, ranked passages for a query.
    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<Scored>> {
        self.search_with(query, k, Filters::default()).await
    }

    /// [`Library::search`] with explicit retrieval filters (trust floor, tags).
    pub async fn search_with(
        &self,
        query: &str,
        k: usize,
        filters: Filters,
    ) -> Result<Vec<Scored>> {
        self.engine.search(query, k, filters).await
    }

    /// Blocking: gather evidence and return a fully resolved [`Answer`].
    pub async fn ask(&self, question: &str) -> Result<Answer> {
        self.ask_with(question, Filters::default()).await
    }

    /// [`Library::ask`] with explicit retrieval filters.
    pub async fn ask_with(&self, question: &str, filters: Filters) -> Result<Answer> {
        agent::ask(
            &self.llm,
            &self.model,
            &self.system,
            question,
            &self.engine,
            self.top_k,
            self.max_rounds,
            filters,
        )
        .await
    }

    /// Streaming: `Search` events while gathering, then `Token`s, then `Done`.
    pub fn stream(&self, question: &str) -> impl Stream<Item = AgentEvent> {
        self.stream_with(question, Filters::default())
    }

    /// [`Library::stream`] with explicit retrieval filters.
    pub fn stream_with(&self, question: &str, filters: Filters) -> impl Stream<Item = AgentEvent> {
        agent::stream(
            self.llm.clone(),
            self.model.clone(),
            self.system.clone(),
            question.to_string(),
            self.engine.clone(),
            self.top_k,
            self.max_rounds,
            filters,
        )
    }

    /// Ingest living content into a collection (upsert, idempotent by doc id).
    pub async fn ingest(&self, collection: &str, docs: Vec<Document>) -> Result<()> {
        let mut store = LocalStore::open(&self.cache_dir, collection, self.embedder.clone());
        store.upsert(docs).await
    }

    /// Remove documents (by id) from a collection — the write-side counterpart of
    /// [`Library::ingest`].
    pub async fn delete(&self, collection: &str, doc_ids: &[String]) -> Result<()> {
        let mut store = LocalStore::open(&self.cache_dir, collection, self.embedder.clone());
        store.delete(doc_ids).await
    }
}

/// Builder returned by [`Library::builder`] for a library over caller-provided
/// parts. Required parts are passed to `builder(...)`; the rest have defaults.
pub struct LibraryBuilder {
    engine: Arc<SearchEngine>,
    embedder: Arc<dyn Embedder>,
    llm: Client,
    model: String,
    system: String,
    cache_dir: PathBuf,
    top_k: usize,
    max_rounds: usize,
}

impl LibraryBuilder {
    /// System prompt (see [`crate::prompt::system_prompt`]). Default: empty.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = system.into();
        self
    }
    /// Where `ingest` / `delete` persist collections. Default: `.cache`.
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }
    /// Passages fed to the agent per search. Default: 5.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }
    /// Max gather rounds. Default: 6.
    pub fn max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n;
        self
    }
    pub fn build(self) -> Library {
        Library {
            llm: self.llm,
            model: self.model,
            system: self.system,
            engine: self.engine,
            cache_dir: self.cache_dir,
            embedder: self.embedder,
            top_k: self.top_k,
            max_rounds: self.max_rounds,
        }
    }
}

/// Resolve the configured lexical backend for a collection. Runtime selection
/// (`ENKI_LEXICAL`); errors if the backend is not compiled in — the
/// features-are-compile-time / config-is-runtime contract.
fn lexical_retriever(cfg: &Config, name: &str) -> Result<Option<Arc<dyn Retriever>>> {
    match cfg.retrieval.lexical.as_str() {
        "none" | "" => Ok(None),
        "tantivy" => {
            #[cfg(feature = "tantivy")]
            {
                let t = crate::lexical::Tantivy::open(&cfg.retrieval.cache_dir, name)?;
                Ok(Some(Arc::new(t) as Arc<dyn Retriever>))
            }
            #[cfg(not(feature = "tantivy"))]
            {
                let _ = name;
                anyhow::bail!("ENKI_LEXICAL=tantivy but the `tantivy` feature is not compiled in")
            }
        }
        other => {
            anyhow::bail!("unknown ENKI_LEXICAL backend `{other}` (expected `none` or `tantivy`)")
        }
    }
}

/// Resolve the configured reranker (second stage). Same compile-time/runtime
/// contract as [`lexical_retriever`].
fn reranker(cfg: &Config) -> Result<Option<Arc<dyn Reranker>>> {
    match cfg.retrieval.rerank.as_str() {
        "none" | "" => Ok(None),
        "fastembed" => {
            #[cfg(feature = "fastembed")]
            {
                let r = crate::rerank::FastembedReranker::open(cfg.retrieval.rerank_cache.clone())?;
                Ok(Some(Arc::new(r) as Arc<dyn Reranker>))
            }
            #[cfg(not(feature = "fastembed"))]
            {
                anyhow::bail!(
                    "ENKI_RERANK=fastembed but the `fastembed` feature is not compiled in"
                )
            }
        }
        other => {
            anyhow::bail!("unknown ENKI_RERANK backend `{other}` (expected `none` or `fastembed`)")
        }
    }
}
