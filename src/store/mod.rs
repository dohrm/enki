//! Persistent, library-owned store for one collection. Two producers feed it via
//! the [`Index`] write interface: the manifest importer (corpus) and the calling
//! application (living "instance" / expert content). One persistence model:
//!
//!   <name>.docs.json   the chunks (id, text, tier, trust, tags, provenance), ordered
//!   <name>.vecs        raw [rows × dim] f32, little-endian, row-aligned with .docs
//!
//! Split across: [`chunk`] (Document → Chunks), [`local`] (write, [`LocalStore`]),
//! [`vector`] (read, [`VectorStore`]).

mod chunk;
mod local;
mod vector;

pub use local::LocalStore;
pub use vector::VectorStore;

// Chunking is shared by every write backend (local files, Qdrant, …). The local
// store reaches it directly; other backends use this re-export.
#[cfg(feature = "qdrant")]
pub(crate) use chunk::chunk_document;

use crate::model::{Chunk, Document};
use anyhow::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};

/// Write interface. Producers push documents; the store chunks, embeds and persists.
#[async_trait]
pub trait Index {
    async fn upsert(&mut self, docs: Vec<Document>) -> Result<()>;
    async fn delete(&mut self, doc_ids: &[String]) -> Result<()>;
}

// ---------- shared paths & (de)serialization (crate-internal) ----------

pub(crate) fn paths(cache_dir: &Path, name: &str) -> (PathBuf, PathBuf) {
    let stem = name.replace([':', '/'], "_");
    (
        cache_dir.join(format!("{stem}.docs.json")),
        cache_dir.join(format!("{stem}.vecs")),
    )
}

pub(crate) fn floats_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(crate) fn floats_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

pub(crate) fn load_docs(path: &Path) -> Vec<Chunk> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::Embedder;
    use crate::model::{Document, Metadata};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Deterministic, offline embedder that counts how many texts it embedded — so
    /// tests can assert incremental upsert re-embeds only what changed.
    struct MockEmbedder {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(texts.len(), Ordering::SeqCst);
            // 3-dim vector derived from the text — enough to round-trip.
            Ok(texts
                .iter()
                .map(|t| {
                    let n = t.chars().count() as f32;
                    vec![
                        n,
                        t.chars().next().map(|c| c as u32 as f32).unwrap_or(0.0),
                        1.0,
                    ]
                })
                .collect())
        }
    }

    fn doc(id: &str, content: &str) -> Document {
        Document {
            id: id.to_string(),
            content: content.to_string(),
            metadata: Metadata {
                label: format!("Doc: {id}"),
                ..Default::default()
            },
        }
    }

    #[tokio::test]
    async fn roundtrip_persist_reload_and_incremental_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder = Arc::new(MockEmbedder {
            calls: calls.clone(),
        });

        // First upsert: two docs → two chunks embedded.
        let mut store = LocalStore::open(dir.path(), "c", embedder.clone());
        store
            .upsert(vec![doc("a", "alpha"), doc("b", "beta")])
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        // Reload read side: chunks + vectors round-trip.
        let vs = VectorStore::open(dir.path(), "c").unwrap();
        let rows: Vec<_> = vs.rows().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].1.len(), 3); // dim preserved

        // Re-upsert unchanged docs: nothing re-embedded (incremental reuse).
        let mut store = LocalStore::open(dir.path(), "c", embedder.clone());
        store
            .upsert(vec![doc("a", "alpha"), doc("b", "beta")])
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "unchanged text must not re-embed"
        );

        // Change one doc's text: exactly one re-embed.
        store.upsert(vec![doc("a", "alpha changed")]).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);

        // Delete one doc: read side shrinks.
        store.delete(&["b".to_string()]).await.unwrap();
        let vs = VectorStore::open(dir.path(), "c").unwrap();
        assert_eq!(vs.rows().count(), 1);
    }
}
