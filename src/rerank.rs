//! Cross-encoder reranking (second stage) via [`fastembed`] — bge-reranker-v2-m3
//! (ONNX, local). Where dense/lexical retrieval scores each passage independently,
//! a cross-encoder scores the (query, passage) pair jointly → sharper ordering of
//! the fused candidate pool before it is cut to the final `k`.
//!
//! Heavy dep (onnxruntime). The model (~2 GB) is fetched to the cache dir on first
//! use; point `cache_dir` at an existing fastembed cache to reuse a download.

use crate::model::{Chunk, Scored};
use crate::search::Reranker;
use anyhow::{Context, Result};
use async_trait::async_trait;
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub struct FastembedReranker {
    model: Arc<Mutex<TextRerank>>,
}

impl FastembedReranker {
    /// Load bge-reranker-v2-m3. `cache_dir` reuses an existing fastembed/HF cache
    /// (e.g. a host app's model dir) instead of downloading; `None` = default.
    pub fn open(cache_dir: Option<PathBuf>) -> Result<Self> {
        let mut opts = RerankInitOptions::new(RerankerModel::BGERerankerV2M3)
            .with_show_download_progress(true);
        if let Some(dir) = cache_dir {
            opts = opts.with_cache_dir(dir);
        }
        let model = TextRerank::try_new(opts).context("loading bge-reranker-v2-m3")?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }
}

#[async_trait]
impl Reranker for FastembedReranker {
    async fn rerank(&self, query: &str, candidates: Vec<Scored>, k: usize) -> Result<Vec<Scored>> {
        if candidates.is_empty() {
            return Ok(candidates);
        }
        let docs: Vec<String> = candidates.iter().map(|s| s.chunk.text.clone()).collect();
        let model = self.model.clone();
        let query = query.to_string();

        // ONNX inference is blocking and `rerank` needs `&mut` — run it off the
        // async worker, behind the mutex.
        let results = tokio::task::spawn_blocking(move || {
            let mut model = model
                .lock()
                .map_err(|_| anyhow::anyhow!("reranker mutex poisoned"))?;
            model.rerank(query, docs, false, None)
        })
        .await
        .context("reranker task")??;

        // Results are sorted by score desc; `index` points back into `candidates`.
        let mut chunks: Vec<Option<Chunk>> =
            candidates.into_iter().map(|s| Some(s.chunk)).collect();
        let mut out = Vec::with_capacity(k.min(results.len()));
        for r in results.into_iter().take(k) {
            if let Some(chunk) = chunks.get_mut(r.index).and_then(Option::take) {
                out.push(Scored {
                    chunk,
                    score: r.score,
                });
            }
        }
        Ok(out)
    }
}
