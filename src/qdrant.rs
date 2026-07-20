//! Qdrant store backend (feature `qdrant`). One profile alongside the local
//! files: a Qdrant server used as both the write target ([`QdrantStore`], an
//! [`Index`]) and a retrieval source ([`QdrantRetriever`], a dense [`Retriever`]).
//!
//! Dense-only for now: candidates come from HNSW over the dense vector, with
//! `trust_min` pushed down as a server-side payload filter. The engine's fusion /
//! rerank stack sits on top unchanged — so the day we add sparse vectors, this
//! backend simply upgrades to [`Modality::Hybrid`] without touching the engine.
//!
//! Mapping:
//! - point id  = `uuid5(chunk_id)` — deterministic, so re-ingest overwrites in place
//! - payload   = the [`Chunk`] as JSON + a numeric `trust_rank` (for range filters)
//! - a document's chunks all carry `doc_id`, so delete/replace is a server filter

use crate::embed::Embedder;
use crate::model::{Chunk, Document, Scored};
use crate::retrieval::{Modality, Query, Retriever};
use crate::store::{Index, chunk_document};
use anyhow::{Context, Result};
use async_trait::async_trait;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
    QueryPointsBuilder, Range, UpsertPointsBuilder, Value, VectorParamsBuilder,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Embedding batch size for ingestion — matches the local store.
const EMBED_BATCH: usize = 16;

/// Connect a Qdrant gRPC client. `api_key` is injected by the caller (Qdrant
/// Cloud); `None` for a local/unsecured instance.
pub fn connect(url: &str, api_key: Option<&str>) -> Result<Qdrant> {
    // Skip the client/server version check: a library must not dictate which
    // Qdrant version the host runs. Our operations (upsert/query/delete) are on
    // the stable surface.
    let mut builder = Qdrant::from_url(url).skip_compatibility_check();
    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        builder = builder.api_key(key.to_string());
    }
    builder.build().context("connecting to Qdrant")
}

/// Deterministic point id from a `chunk_id` — so re-upsert overwrites the same
/// point rather than creating a duplicate. Qdrant point ids must be `u64` or a
/// UUID; our ids are arbitrary strings, hence the v5 hash.
fn point_id(chunk_id: &str) -> String {
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, chunk_id.as_bytes()).to_string()
}

/// A [`Chunk`] as a Qdrant payload: its JSON fields plus a numeric `trust_rank`
/// so `trust_min` can be a server-side range filter (the enum can't be ranked).
fn payload_of(chunk: &Chunk) -> Result<Payload> {
    let mut value = serde_json::to_value(chunk)?;
    if let serde_json::Value::Object(map) = &mut value {
        map.insert("trust_rank".into(), serde_json::json!(chunk.trust.rank()));
    }
    Payload::try_from(value).context("building Qdrant payload")
}

/// Reconstruct a [`Chunk`] from a retrieved payload (the extra `trust_rank` is
/// ignored by `Chunk`'s deserializer).
fn chunk_from_payload(payload: HashMap<String, Value>) -> Result<Chunk> {
    let map: serde_json::Map<String, serde_json::Value> = payload
        .into_iter()
        .map(|(k, v)| (k, v.into_json()))
        .collect();
    serde_json::from_value(serde_json::Value::Object(map)).context("decoding chunk from payload")
}

// ---------- Write side ----------

/// A Qdrant-backed collection implementing [`Index`]. The collection is created
/// lazily on first upsert (dimension inferred from the embeddings).
pub struct QdrantStore {
    client: Qdrant,
    name: String,
    embedder: Arc<dyn Embedder>,
}

impl QdrantStore {
    pub fn open(
        url: &str,
        api_key: Option<&str>,
        name: &str,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self> {
        Ok(Self {
            client: connect(url, api_key)?,
            name: name.to_string(),
            embedder,
        })
    }

    async fn ensure_collection(&self, dim: u64) -> Result<()> {
        if !self.client.collection_exists(&self.name).await? {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.name)
                        .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
                )
                .await
                .with_context(|| format!("creating Qdrant collection `{}`", self.name))?;
        }
        Ok(())
    }

    /// Drop every point belonging to the given documents (replace semantics) —
    /// one server-side filter on `doc_id`. No-op if the collection is absent.
    async fn delete_docs(&self, doc_ids: &[String]) -> Result<()> {
        if doc_ids.is_empty() || !self.client.collection_exists(&self.name).await? {
            return Ok(());
        }
        let conds: Vec<Condition> = doc_ids
            .iter()
            .map(|id| Condition::matches("doc_id", id.clone()))
            .collect();
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.name)
                    .points(Filter::should(conds))
                    .wait(true),
            )
            .await
            .with_context(|| format!("deleting from Qdrant collection `{}`", self.name))?;
        Ok(())
    }
}

#[async_trait]
impl Index for QdrantStore {
    async fn upsert(&mut self, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let chunks: Vec<Chunk> = docs.iter().flat_map(chunk_document).collect();
        if chunks.is_empty() {
            return Ok(());
        }

        // Embed every chunk of the touched documents (they are re-written wholesale).
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(EMBED_BATCH) {
            let texts: Vec<String> = batch.iter().map(|c| c.embed_input()).collect();
            vectors.extend(self.embedder.embed(&texts).await?);
        }
        let dim = vectors.first().map(Vec::len).context("empty embedding")? as u64;

        // Create → clear old versions → write. Order matters: delete_docs needs the
        // collection to exist.
        self.ensure_collection(dim).await?;
        let doc_ids: Vec<String> = docs.iter().map(|d| d.id.clone()).collect();
        self.delete_docs(&doc_ids).await?;

        let mut points = Vec::with_capacity(chunks.len());
        for (chunk, vec) in chunks.iter().zip(vectors) {
            points.push(PointStruct::new(
                point_id(&chunk.chunk_id),
                vec,
                payload_of(chunk)?,
            ));
        }
        let count = points.len();
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.name, points).wait(true))
            .await
            .with_context(|| format!("upserting into Qdrant collection `{}`", self.name))?;
        tracing::debug!(target: "enki", collection = %self.name, count, "qdrant upsert");
        Ok(())
    }

    async fn delete(&mut self, doc_ids: &[String]) -> Result<()> {
        self.delete_docs(doc_ids).await
    }
}

// ---------- Read side ----------

/// A dense retrieval source over one Qdrant collection. Declares
/// [`Modality::Dense`]; the query vector is supplied by the engine.
pub struct QdrantRetriever {
    client: Arc<Qdrant>,
    name: String,
}

impl QdrantRetriever {
    pub fn new(client: Arc<Qdrant>, name: impl Into<String>) -> Self {
        Self {
            client,
            name: name.into(),
        }
    }
}

#[async_trait]
impl Retriever for QdrantRetriever {
    fn modality(&self) -> Modality {
        Modality::Dense
    }

    async fn retrieve(&self, q: &Query) -> Result<Vec<Scored>> {
        let dense = q
            .dense
            .as_deref()
            .context("qdrant retriever requires a dense query vector")?;

        let mut builder = QueryPointsBuilder::new(&self.name)
            .query(dense.to_vec())
            .limit(q.k as u64)
            .with_payload(true);
        // Push `trust_min` down to the server as a range on the numeric rank.
        if let Some(min) = q.filters.trust_min {
            builder = builder.filter(Filter::must([Condition::range(
                "trust_rank",
                Range {
                    gte: Some(min.rank() as f64),
                    ..Default::default()
                },
            )]));
        }

        let res = self
            .client
            .query(builder)
            .await
            .with_context(|| format!("querying Qdrant collection `{}`", self.name))?;

        Ok(res
            .result
            .into_iter()
            .filter_map(|p| {
                let score = p.score;
                chunk_from_payload(p.payload)
                    .ok()
                    .map(|chunk| Scored { chunk, score })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Provenance, TrustStatus};

    fn chunk() -> Chunk {
        Chunk {
            chunk_id: "spell:boule-de-feu#1".into(),
            doc_id: "spell:boule-de-feu".into(),
            text: "8d6 dégâts de feu".into(),
            tier: 0,
            trust: TrustStatus::Canonical,
            tags: vec!["mage".into()],
            provenance: Provenance {
                source: "SRD".into(),
                label: "Sorts > Boule de feu".into(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn point_id_is_deterministic_and_uuid() {
        let a = point_id("spell:x#0");
        let b = point_id("spell:x#0");
        assert_eq!(a, b, "same chunk_id → same point id (idempotent upsert)");
        assert_ne!(a, point_id("spell:x#1"));
        assert!(uuid::Uuid::parse_str(&a).is_ok(), "must be a valid UUID");
    }

    #[test]
    fn payload_roundtrips_through_chunk() {
        let c = chunk();
        let payload = payload_of(&c).unwrap();
        // Payload derefs to the same map shape a retrieved point carries.
        let map: HashMap<String, Value> = payload.into();
        assert!(map.contains_key("trust_rank"), "rank exposed for filtering");
        let back = chunk_from_payload(map).unwrap();
        assert_eq!(back.chunk_id, c.chunk_id);
        assert_eq!(back.doc_id, c.doc_id);
        assert_eq!(back.text, c.text);
        assert_eq!(back.trust, c.trust);
        assert_eq!(back.tags, c.tags);
        assert_eq!(back.provenance.label, c.provenance.label);
    }

    /// Live round-trip against a real Qdrant. Skipped unless `ENKI_QDRANT_TEST=1`
    /// (so CI, which has no server, is green). Uses a deterministic offline
    /// embedder and a throwaway collection it cleans up.
    #[tokio::test]
    async fn live_upsert_query_delete() {
        if std::env::var("ENKI_QDRANT_TEST").as_deref() != Ok("1") {
            eprintln!("skipping: set ENKI_QDRANT_TEST=1 with a running Qdrant to run");
            return;
        }
        use crate::model::Metadata;

        struct Mock;
        #[async_trait]
        impl Embedder for Mock {
            async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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

        let url = std::env::var("ENKI_QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());
        let name = "enki_test_live";
        let embedder: Arc<dyn Embedder> = Arc::new(Mock);
        let client = Arc::new(connect(&url, None).unwrap());
        client.delete_collection(name).await.ok();

        let doc = Document {
            id: "d1".into(),
            content: "alpha beta gamma".into(),
            metadata: Metadata {
                label: "Doc d1".into(),
                trust: TrustStatus::Canonical,
                ..Default::default()
            },
        };
        let mut store = QdrantStore::open(&url, None, name, embedder.clone()).unwrap();
        store.upsert(vec![doc]).await.unwrap();

        let retriever = QdrantRetriever::new(client.clone(), name);
        let dense = embedder.embed(&["alpha beta gamma".into()]).await.unwrap();
        let q = Query {
            text: "alpha beta gamma".into(),
            dense: Some(dense.into_iter().next().unwrap()),
            k: 5,
            filters: Default::default(),
        };
        let hits = retriever.retrieve(&q).await.unwrap();
        assert!(!hits.is_empty(), "should retrieve the ingested chunk");
        assert_eq!(hits[0].chunk.doc_id, "d1");

        store.delete(&["d1".to_string()]).await.unwrap();
        let after = retriever.retrieve(&q).await.unwrap();
        assert!(after.is_empty(), "delete removed the document's points");

        client.delete_collection(name).await.ok();
    }
}
