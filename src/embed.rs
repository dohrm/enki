//! Embeddings via genai. Persistence lives in `store`; the indexing pipeline
//! drives incremental embedding.

use anyhow::{Context, Result};
use async_trait::async_trait;
use genai::Client;

/// Provider-agnostic embedding seam. POC impl: Ollama via genai.
/// `async_trait` so it can be used behind `Arc<dyn Embedder>` in the search engine.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub struct GenaiEmbedder {
    client: Client,
    model: String,
}

impl GenaiEmbedder {
    pub fn new(client: Client, model: String) -> Self {
        Self { client, model }
    }
}

#[async_trait]
impl Embedder for GenaiEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let t = std::time::Instant::now();
        let res = self
            .client
            .embed_batch(self.model.as_str(), texts.to_vec(), None)
            .await
            .context("embeddings call")?;
        let vectors = res.into_vectors();
        tracing::debug!(
            target: "enki",
            model = %self.model,
            count = vectors.len(),
            elapsed_ms = t.elapsed().as_millis(),
            "embed batch"
        );
        Ok(vectors)
    }
}
