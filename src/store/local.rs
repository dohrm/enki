//! Write side: a filesystem-backed collection implementing [`Index`]. Held in RAM
//! during writes and persisted on every mutation. Upsert is incremental — only
//! new or text-changed chunks are re-embedded.

use super::chunk::chunk_document;
use super::{Index, floats_from_bytes, floats_to_bytes, load_docs, paths};
use crate::embed::Embedder;
use crate::model::{Chunk, Document};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct LocalStore {
    cache_dir: PathBuf,
    name: String,
    embedder: Arc<dyn Embedder>,
    chunks: Vec<Chunk>,
    vecs: Vec<f32>, // row-aligned with `chunks`
    dim: usize,
}

impl LocalStore {
    /// Open (or start empty) the collection `name` under `cache_dir`.
    pub fn open(cache_dir: &Path, name: &str, embedder: Arc<dyn Embedder>) -> Self {
        let (docs_path, vecs_path) = paths(cache_dir, name);
        let chunks = load_docs(&docs_path);
        let vecs = std::fs::read(&vecs_path)
            .map(|b| floats_from_bytes(&b))
            .unwrap_or_default();
        let dim = if chunks.is_empty() {
            0
        } else {
            vecs.len() / chunks.len()
        };
        Self {
            cache_dir: cache_dir.to_path_buf(),
            name: name.to_string(),
            embedder,
            chunks,
            vecs,
            dim,
        }
    }

    fn row(&self, i: usize) -> &[f32] {
        &self.vecs[i * self.dim..(i + 1) * self.dim]
    }

    fn persist(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir).ok();
        let (docs_path, vecs_path) = paths(&self.cache_dir, &self.name);
        std::fs::write(&docs_path, serde_json::to_string(&self.chunks)?)
            .with_context(|| format!("writing {}", docs_path.display()))?;
        std::fs::write(&vecs_path, floats_to_bytes(&self.vecs))
            .with_context(|| format!("writing {}", vecs_path.display()))?;
        // Rebuild the lexical index from the full chunk set — single write path,
        // always in sync with the dense store. No-op unless `tantivy` is compiled.
        #[cfg(feature = "tantivy")]
        crate::lexical::build_index(&self.cache_dir, &self.name, &self.chunks)?;
        Ok(())
    }
}

#[async_trait]
impl Index for LocalStore {
    async fn upsert(&mut self, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }

        // Snapshot existing chunk_id -> (text, vector) so unchanged chunks are reused.
        let old: HashMap<String, (String, Vec<f32>)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.chunk_id.clone(), (c.text.clone(), self.row(i).to_vec())))
            .collect();

        // Drop chunks belonging to the upserted documents (replace semantics).
        let upsert_ids: HashSet<&str> = docs.iter().map(|d| d.id.as_str()).collect();
        let mut chunks: Vec<Chunk> = std::mem::take(&mut self.chunks)
            .into_iter()
            .filter(|c| !upsert_ids.contains(c.doc_id.as_str()))
            .collect();
        chunks.extend(docs.iter().flat_map(chunk_document));

        // Embed only chunks that are new or whose text changed.
        let to_embed: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| match old.get(&c.chunk_id) {
                Some((text, _)) => text != &c.text,
                None => true,
            })
            .collect();
        let mut fresh: HashMap<String, Vec<f32>> = HashMap::new();
        if !to_embed.is_empty() {
            println!("embedding: {} chunks ({})", to_embed.len(), self.name);
            for batch in to_embed.chunks(16) {
                let texts: Vec<String> = batch.iter().map(|c| c.embed_input()).collect();
                let vectors = self.embedder.embed(&texts).await?;
                for (c, v) in batch.iter().zip(vectors) {
                    fresh.insert(c.chunk_id.clone(), v);
                }
            }
        }

        // Reassemble the vector matrix aligned with `chunks`.
        let mut vecs: Vec<f32> = Vec::new();
        for c in &chunks {
            let v = fresh
                .get(&c.chunk_id)
                .or_else(|| old.get(&c.chunk_id).map(|(_, v)| v))
                .with_context(|| format!("no vector for {}", c.chunk_id))?;
            vecs.extend_from_slice(v);
        }

        self.dim = vecs.len().checked_div(chunks.len()).unwrap_or(0);
        self.chunks = chunks;
        self.vecs = vecs;
        self.persist()
    }

    async fn delete(&mut self, doc_ids: &[String]) -> Result<()> {
        let drop: HashSet<&str> = doc_ids.iter().map(String::as_str).collect();
        let keep: Vec<usize> = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| !drop.contains(c.doc_id.as_str()))
            .map(|(i, _)| i)
            .collect();
        let mut vecs = Vec::with_capacity(keep.len() * self.dim);
        for &i in &keep {
            vecs.extend_from_slice(self.row(i));
        }
        self.chunks = keep.iter().map(|&i| self.chunks[i].clone()).collect();
        self.vecs = vecs;
        self.persist()
    }
}
