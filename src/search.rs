//! Search service. Composes retrievers into a two-axis fusion, then an optional
//! rerank:
//!
//! 1. modality fusion (RRF) within a collection — relevance signal
//! 2. tier fusion across collections — authority signal
//!
//! Backends are hidden behind `Retriever`; a hybrid backend is just a collection
//! with a single retriever.

use crate::embed::Embedder;
use crate::model::{Chunk, Scored, Tier};
use crate::retrieval::{Filters, Query, Retriever};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

// ---------- Fusion ----------

pub trait Fuser: Send + Sync {
    fn fuse(&self, lists: Vec<Vec<Scored>>, k: usize) -> Vec<Scored>;
}

/// Reciprocal Rank Fusion. Rank-based, so it fuses heterogeneous sources whose
/// raw scores are not comparable (different models / collections).
pub struct Rrf {
    pub k0: f32,
}

impl Default for Rrf {
    fn default() -> Self {
        Self { k0: 60.0 }
    }
}

impl Fuser for Rrf {
    fn fuse(&self, lists: Vec<Vec<Scored>>, k: usize) -> Vec<Scored> {
        let mut acc: HashMap<String, (Chunk, f32)> = HashMap::new();
        for list in lists {
            for (rank, scored) in list.into_iter().enumerate() {
                let contrib = 1.0 / (self.k0 + rank as f32 + 1.0);
                match acc.get_mut(&scored.chunk.chunk_id) {
                    Some(entry) => entry.1 += contrib,
                    None => {
                        let id = scored.chunk.chunk_id.clone();
                        acc.insert(id, (scored.chunk, contrib));
                    }
                }
            }
        }
        let mut out: Vec<Scored> = acc
            .into_values()
            .map(|(chunk, score)| Scored { chunk, score })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(k);
        out
    }
}

// ---------- Rerank (optional) ----------

/// Candidate multiplier: with a reranker, fetch `k * RERANK_POOL` before it cuts
/// back to `k`, so it has passages to actually reorder.
const RERANK_POOL: usize = 4;

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, candidates: Vec<Scored>, k: usize) -> Result<Vec<Scored>>;
}

// ---------- Collections ----------

/// A collection = one index/tier and its retrievers (its modalities). Qdrant
/// hybrid → 1 retriever; local profile → brute-force + tantivy → 2 retrievers.
pub struct Collection {
    pub name: String,
    pub tier: Tier,
    pub retrievers: Vec<Arc<dyn Retriever>>,
}

// ---------- Engine ----------

pub struct SearchEngine {
    embedder: Arc<dyn Embedder>,
    collections: Vec<Collection>,
    modality_fuser: Box<dyn Fuser>,
    tier_fuser: Box<dyn Fuser>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl SearchEngine {
    pub fn new(embedder: Arc<dyn Embedder>, collections: Vec<Collection>) -> Self {
        Self {
            embedder,
            collections,
            modality_fuser: Box::new(Rrf::default()),
            // NOTE: plain RRF for now. Authority weighting (tier-boost) lands once
            // we have reference benchmark data — no premature tuning.
            tier_fuser: Box::new(Rrf::default()),
            reranker: None,
        }
    }

    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }

    #[tracing::instrument(target = "enki", skip(self, filters), fields(k, hits))]
    pub async fn search(&self, text: &str, k: usize, filters: Filters) -> Result<Vec<Scored>> {
        let started = std::time::Instant::now();
        // 1. Embed the query once, only if some source needs a dense vector.
        let needs_dense = self
            .collections
            .iter()
            .flat_map(|c| &c.retrievers)
            .any(|r| r.modality().needs_dense());
        let dense = if needs_dense {
            let t = std::time::Instant::now();
            let v = self
                .embedder
                .embed(&[text.to_string()])
                .await?
                .into_iter()
                .next()
                .context("empty query embedding")?;
            tracing::debug!(target: "enki", dim = v.len(), elapsed_ms = t.elapsed().as_millis(), "embed query");
            Some(v)
        } else {
            None
        };
        // A reranker needs a candidate pool larger than the final `k` to reorder;
        // retrieve/fuse wider, then let it cut down to `k`.
        let cand_k = if self.reranker.is_some() {
            (k * RERANK_POOL).max(k)
        } else {
            k
        };
        let query = Query {
            text: text.to_string(),
            dense,
            k: cand_k,
            filters,
        };

        // 2. Per collection: run its retrievers concurrently, fuse modalities.
        let mut per_collection: Vec<Vec<Scored>> = Vec::with_capacity(self.collections.len());
        for collection in &self.collections {
            let lists = futures::future::try_join_all(
                collection.retrievers.iter().map(|r| r.retrieve(&query)),
            )
            .await?;
            per_collection.push(self.modality_fuser.fuse(lists, cand_k));
        }

        // 3. Fuse across collections (authority axis). Identity for a single collection.
        let mut fused = self.tier_fuser.fuse(per_collection, cand_k);

        // 4. Optional rerank down to the final k.
        if let Some(reranker) = &self.reranker {
            fused = reranker.rerank(text, fused, k).await?;
        }

        fused.truncate(k);
        tracing::Span::current().record("hits", fused.len());
        tracing::debug!(
            target: "enki",
            hits = fused.len(),
            collections = self.collections.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "search done"
        );
        Ok(fused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Provenance;

    fn scored(id: &str, score: f32) -> Scored {
        Scored {
            chunk: Chunk {
                chunk_id: id.into(),
                doc_id: id.into(),
                text: id.into(),
                tier: 0,
                trust: crate::model::TrustStatus::Draft,
                tags: vec![],
                provenance: Provenance::default(),
            },
            score,
        }
    }

    #[test]
    fn rrf_rewards_agreement_across_lists_and_dedups() {
        // `b` is high in both lists; `a` only tops list 1; `c` only tops list 2.
        let list1 = vec![scored("a", 9.0), scored("b", 8.0)];
        let list2 = vec![scored("b", 9.0), scored("c", 1.0)];
        let out = Rrf::default().fuse(vec![list1, list2], 10);

        // No duplicates by chunk_id.
        assert_eq!(out.len(), 3);
        // `b` (ranked in both) wins over singly-ranked `a`/`c`.
        assert_eq!(out[0].chunk.chunk_id, "b");
    }

    #[test]
    fn rrf_truncates_to_k() {
        let list = vec![scored("a", 3.0), scored("b", 2.0), scored("c", 1.0)];
        let out = Rrf::default().fuse(vec![list], 2);
        assert_eq!(out.len(), 2);
    }
}
