//! Retrieval backends (the "sources of candidates"). A `Retriever` produces a
//! ranked candidate list for one query; it may serve one modality (dense OR
//! lexical) or both (a hybrid backend like Qdrant). Fusion happens above, in the
//! search engine — so a hybrid backend is simply a single source, never a noop.

use crate::model::{Chunk, Scored, TrustStatus};
use crate::store::VectorStore;
use anyhow::{Context, Result};
use async_trait::async_trait;

/// What a retriever covers. A hybrid backend declares `Hybrid` and is treated as
/// one source; the engine never splits it into a dense + lexical pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Modality {
    Dense,
    Lexical,
    Hybrid,
}

impl Modality {
    pub fn needs_dense(self) -> bool {
        matches!(self, Modality::Dense | Modality::Hybrid)
    }
    pub fn needs_lexical(self) -> bool {
        matches!(self, Modality::Lexical | Modality::Hybrid)
    }
}

/// Search-time filters (dynamic, fine-grained). Tier partitioning is handled by
/// which collections are queried, not here.
#[derive(Default, Clone)]
pub struct Filters {
    /// Minimum trust required (e.g. `Endorsed` to exclude unvetted expert notes).
    pub trust_min: Option<TrustStatus>,
    /// Reserved: tag filtering once chunks carry tags.
    pub tags: Vec<String>,
}

impl Filters {
    pub fn allows(&self, chunk: &Chunk) -> bool {
        match self.trust_min {
            Some(min) => chunk.trust.rank() >= min.rank(),
            None => true,
        }
    }
}

/// One query, embedded once by the engine and fanned out to every retriever.
pub struct Query {
    pub text: String,
    /// Dense embedding, filled by the engine when any source needs it.
    pub dense: Option<Vec<f32>>,
    pub k: usize,
    pub filters: Filters,
}

#[async_trait]
pub trait Retriever: Send + Sync {
    fn modality(&self) -> Modality;
    async fn retrieve(&self, q: &Query) -> Result<Vec<Scored>>;

    /// Fetch every chunk of the given documents **by id** — key access, not
    /// similarity (used to turn graph neighbours / handles into passages). The
    /// returned `score` is not a relevance signal. Default: unsupported (empty);
    /// content-holding backends override it.
    async fn fetch(&self, _doc_ids: &[String]) -> Result<Vec<Scored>> {
        Ok(Vec::new())
    }
}

// ---------- Dense brute-force (local profile, zero infra) ----------

pub struct BruteForce {
    store: VectorStore,
}

impl BruteForce {
    pub fn new(store: VectorStore) -> Self {
        Self { store }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[async_trait]
impl Retriever for BruteForce {
    fn modality(&self) -> Modality {
        Modality::Dense
    }

    async fn retrieve(&self, q: &Query) -> Result<Vec<Scored>> {
        let qv = q
            .dense
            .as_deref()
            .context("brute-force retriever requires a dense query vector")?;

        let mut scored: Vec<Scored> = self
            .store
            .rows()
            .filter_map(|(chunk, vec)| {
                q.filters.allows(chunk).then(|| Scored {
                    chunk: chunk.clone(),
                    score: cosine(qv, vec),
                })
            })
            .collect();
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(q.k);
        Ok(scored)
    }

    async fn fetch(&self, doc_ids: &[String]) -> Result<Vec<Scored>> {
        let want: std::collections::HashSet<&str> = doc_ids.iter().map(String::as_str).collect();
        Ok(self
            .store
            .rows()
            .filter(|(chunk, _)| want.contains(chunk.doc_id.as_str()))
            .map(|(chunk, _)| Scored {
                chunk: chunk.clone(),
                score: 1.0,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Chunk, Provenance};

    #[test]
    fn cosine_is_1_for_identical_and_0_for_orthogonal() {
        assert!((cosine(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0); // zero vector → 0, no NaN
    }

    fn chunk(trust: TrustStatus) -> Chunk {
        Chunk {
            chunk_id: "c#0".into(),
            doc_id: "c".into(),
            text: "t".into(),
            tier: 0,
            trust,
            tags: vec![],
            provenance: Provenance::default(),
        }
    }

    #[test]
    fn filters_gate_on_trust_min() {
        let f = Filters {
            trust_min: Some(TrustStatus::Endorsed),
            tags: vec![],
        };
        assert!(f.allows(&chunk(TrustStatus::Canonical)));
        assert!(f.allows(&chunk(TrustStatus::Endorsed)));
        assert!(!f.allows(&chunk(TrustStatus::Draft)));
        // No floor → everything passes.
        assert!(Filters::default().allows(&chunk(TrustStatus::Draft)));
    }
}
