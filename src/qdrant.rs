//! Qdrant store backend (feature `qdrant`). One profile alongside the local
//! files: a Qdrant server used as both the write target ([`QdrantStore`], an
//! [`Index`]) and a retrieval source ([`QdrantRetriever`], a [`Retriever`]).
//!
//! Two modes, chosen by whether a [`SparseEmbedder`] is supplied:
//! - **dense-only** (no sparse): a single unnamed dense vector; `Modality::Dense`.
//! - **hybrid** (dense + sparse): named `dense` + `sparse` vectors, fused
//!   server-side (RRF/DBSF) in one Query API call; `Modality::Hybrid`. The sparse
//!   config carries `Modifier::Idf` so the server applies IDF to the sparse dot
//!   product (turning raw-TF sparse vectors into a BM25-ish signal).
//!
//! Either way the engine sees one source per collection — its fusion/rerank stack
//! is untouched. `trust_min` is pushed down as a server-side payload filter.
//!
//! Mapping:
//! - point id  = `uuid5(chunk_id)` — deterministic, so re-ingest overwrites in place
//! - payload   = the [`Chunk`] as JSON + a numeric `trust_rank` (for range filters)
//! - a document's chunks all carry `doc_id`, so delete/replace is a server filter

use crate::embed::Embedder;
use crate::model::{Chunk, Document, Scored};
use crate::retrieval::{Modality, Query as EngineQuery, Retriever};
use crate::sparse::SparseEmbedder;
use crate::store::{Index, chunk_document};
use anyhow::{Context, Result};
use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, Fusion, Modifier,
    NamedVectors, PointStruct, PrefetchQueryBuilder, Query, QueryPointsBuilder, Range,
    ScrollPointsBuilder, SparseVectorParamsBuilder, SparseVectorsConfigBuilder,
    UpsertPointsBuilder, Value, Vector, VectorInput, VectorParamsBuilder, VectorsConfigBuilder,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Embedding batch size for ingestion — matches the local store.
const EMBED_BATCH: usize = 16;
/// Named vectors used in hybrid collections.
const DENSE: &str = "dense";
const SPARSE: &str = "sparse";
/// Cap on points returned by a by-id `fetch` (graph neighbours / open).
const FETCH_LIMIT: u32 = 512;

/// Connect a Qdrant gRPC client. `api_key` is injected by the caller (Qdrant
/// Cloud); `None` for a local/unsecured instance.
pub fn connect(url: &str, api_key: Option<&str>) -> Result<Qdrant> {
    // Skip the client/server version check: a library must not dictate which
    // Qdrant version the host runs. Our operations are on the stable surface.
    let mut builder = Qdrant::from_url(url).skip_compatibility_check();
    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        builder = builder.api_key(key.to_string());
    }
    builder.build().context("connecting to Qdrant")
}

/// Parse the configured server-side fusion strategy.
pub fn fusion_from_str(s: &str) -> Result<Fusion> {
    match s {
        "rrf" | "" => Ok(Fusion::Rrf),
        "dbsf" => Ok(Fusion::Dbsf),
        other => anyhow::bail!("unknown ENKI_QDRANT_FUSION `{other}` (expected `rrf` or `dbsf`)"),
    }
}

/// Deterministic point id from a `chunk_id` — so re-upsert overwrites the same
/// point rather than creating a duplicate. Qdrant point ids must be `u64` or a
/// UUID; our ids are arbitrary strings, hence the v5 hash.
fn point_id(chunk_id: &str) -> String {
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, chunk_id.as_bytes()).to_string()
}

/// A [`Chunk`] as a Qdrant payload: its JSON fields plus a numeric `trust_rank`
/// so `trust_min` can be a server-side range filter (the enum can't be ranked).
fn payload_of(chunk: &Chunk) -> Result<qdrant_client::Payload> {
    let mut value = serde_json::to_value(chunk)?;
    if let serde_json::Value::Object(map) = &mut value {
        map.insert("trust_rank".into(), serde_json::json!(chunk.trust.rank()));
    }
    qdrant_client::Payload::try_from(value).context("building Qdrant payload")
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

/// Server-side `trust_min` filter (a range on the numeric `trust_rank`).
fn trust_filter(q: &EngineQuery) -> Option<Filter> {
    q.filters.trust_min.map(|min| {
        Filter::must([Condition::range(
            "trust_rank",
            Range {
                gte: Some(min.rank() as f64),
                ..Default::default()
            },
        )])
    })
}

// ---------- Write side ----------

/// A Qdrant-backed collection implementing [`Index`]. The collection is created
/// lazily on first upsert (dimension inferred from the embeddings). Hybrid when a
/// [`SparseEmbedder`] is supplied.
pub struct QdrantStore {
    client: Qdrant,
    name: String,
    embedder: Arc<dyn Embedder>,
    sparse: Option<Arc<dyn SparseEmbedder>>,
}

impl QdrantStore {
    pub fn open(
        url: &str,
        api_key: Option<&str>,
        name: &str,
        embedder: Arc<dyn Embedder>,
        sparse: Option<Arc<dyn SparseEmbedder>>,
    ) -> Result<Self> {
        Ok(Self {
            client: connect(url, api_key)?,
            name: name.to_string(),
            embedder,
            sparse,
        })
    }

    async fn ensure_collection(&self, dim: u64) -> Result<()> {
        if self.client.collection_exists(&self.name).await? {
            return Ok(());
        }
        let create = if self.sparse.is_some() {
            // Hybrid: named dense + sparse (IDF-modified so the server applies IDF).
            let mut dense_cfg = VectorsConfigBuilder::default();
            dense_cfg
                .add_named_vector_params(DENSE, VectorParamsBuilder::new(dim, Distance::Cosine));
            let mut sparse_cfg = SparseVectorsConfigBuilder::default();
            sparse_cfg.add_named_vector_params(
                SPARSE,
                SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
            );
            CreateCollectionBuilder::new(&self.name)
                .vectors_config(dense_cfg)
                .sparse_vectors_config(sparse_cfg)
        } else {
            CreateCollectionBuilder::new(&self.name)
                .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine))
        };
        self.client
            .create_collection(create)
            .await
            .with_context(|| format!("creating Qdrant collection `{}`", self.name))?;
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
        let texts: Vec<String> = chunks.iter().map(|c| c.embed_input()).collect();

        // Dense (batched), + sparse for the whole set when hybrid.
        let mut dense: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        for batch in texts.chunks(EMBED_BATCH) {
            dense.extend(self.embedder.embed(batch).await?);
        }
        let sparse = match &self.sparse {
            Some(s) => Some(s.embed_sparse(&texts).await?),
            None => None,
        };
        let dim = dense.first().map(Vec::len).context("empty embedding")? as u64;

        // Create → clear old versions → write. Order matters: delete_docs needs the
        // collection to exist.
        self.ensure_collection(dim).await?;
        let doc_ids: Vec<String> = docs.iter().map(|d| d.id.clone()).collect();
        self.delete_docs(&doc_ids).await?;

        let mut points = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let payload = payload_of(chunk)?;
            let id = point_id(&chunk.chunk_id);
            let point = match &sparse {
                Some(sv) => {
                    let s = &sv[i];
                    let vectors = NamedVectors::default()
                        .add_vector(DENSE, dense[i].clone())
                        .add_vector(
                            SPARSE,
                            Vector::new_sparse(s.indices.clone(), s.values.clone()),
                        );
                    PointStruct::new(id, vectors, payload)
                }
                None => PointStruct::new(id, dense[i].clone(), payload),
            };
            points.push(point);
        }
        let count = points.len();
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.name, points).wait(true))
            .await
            .with_context(|| format!("upserting into Qdrant collection `{}`", self.name))?;
        tracing::debug!(target: "enki", collection = %self.name, count, hybrid = sparse.is_some(), "qdrant upsert");
        Ok(())
    }

    async fn delete(&mut self, doc_ids: &[String]) -> Result<()> {
        self.delete_docs(doc_ids).await
    }
}

// ---------- Read side ----------

/// A retrieval source over one Qdrant collection. Dense-only, or hybrid
/// (dense+sparse with server-side fusion) when a [`SparseEmbedder`] is supplied.
pub struct QdrantRetriever {
    client: Arc<Qdrant>,
    name: String,
    sparse: Option<Arc<dyn SparseEmbedder>>,
    fusion: Fusion,
}

impl QdrantRetriever {
    /// Dense-only source.
    pub fn dense(client: Arc<Qdrant>, name: impl Into<String>) -> Self {
        Self {
            client,
            name: name.into(),
            sparse: None,
            fusion: Fusion::Rrf,
        }
    }

    /// Hybrid source: dense + sparse fused server-side by `fusion`.
    pub fn hybrid(
        client: Arc<Qdrant>,
        name: impl Into<String>,
        sparse: Arc<dyn SparseEmbedder>,
        fusion: Fusion,
    ) -> Self {
        Self {
            client,
            name: name.into(),
            sparse: Some(sparse),
            fusion,
        }
    }

    async fn retrieve_dense(&self, q: &EngineQuery, dense: &[f32]) -> Result<Vec<Scored>> {
        let mut builder = QueryPointsBuilder::new(&self.name)
            .query(dense.to_vec())
            .limit(q.k as u64)
            .with_payload(true);
        if let Some(f) = trust_filter(q) {
            builder = builder.filter(f);
        }
        let res = self.client.query(builder).await?;
        Ok(scored_from(res.result))
    }

    async fn retrieve_hybrid(
        &self,
        q: &EngineQuery,
        dense: &[f32],
        sparse: &Arc<dyn SparseEmbedder>,
    ) -> Result<Vec<Scored>> {
        let sv = sparse
            .embed_sparse(std::slice::from_ref(&q.text))
            .await?
            .into_iter()
            .next()
            .unwrap_or_default();

        // Each branch retrieves `k` candidates; the server fuses to `k`. Trust is
        // filtered inside each prefetch (server-side).
        let filter = trust_filter(q);
        let mut dense_pf = PrefetchQueryBuilder::default()
            .using(DENSE)
            .query(dense.to_vec())
            .limit(q.k as u64);
        let mut sparse_pf = PrefetchQueryBuilder::default()
            .using(SPARSE)
            .query(Query::new_nearest(VectorInput::new_sparse(
                sv.indices, sv.values,
            )))
            .limit(q.k as u64);
        if let Some(f) = &filter {
            dense_pf = dense_pf.filter(f.clone());
            sparse_pf = sparse_pf.filter(f.clone());
        }

        let builder = QueryPointsBuilder::new(&self.name)
            .add_prefetch(dense_pf)
            .add_prefetch(sparse_pf)
            .query(Query::new_fusion(self.fusion))
            .limit(q.k as u64)
            .with_payload(true);
        let res = self.client.query(builder).await?;
        Ok(scored_from(res.result))
    }
}

/// Map Qdrant scored points → engine `Scored`, dropping any with an undecodable payload.
fn scored_from(points: Vec<qdrant_client::qdrant::ScoredPoint>) -> Vec<Scored> {
    points
        .into_iter()
        .filter_map(|p| {
            let score = p.score;
            chunk_from_payload(p.payload)
                .ok()
                .map(|chunk| Scored { chunk, score })
        })
        .collect()
}

#[async_trait]
impl Retriever for QdrantRetriever {
    fn modality(&self) -> Modality {
        if self.sparse.is_some() {
            Modality::Hybrid
        } else {
            Modality::Dense
        }
    }

    async fn retrieve(&self, q: &EngineQuery) -> Result<Vec<Scored>> {
        let dense = q
            .dense
            .as_deref()
            .context("qdrant retriever requires a dense query vector")?;
        match &self.sparse {
            Some(sparse) => self.retrieve_hybrid(q, dense, sparse).await,
            None => self.retrieve_dense(q, dense).await,
        }
        .with_context(|| format!("querying Qdrant collection `{}`", self.name))
    }

    async fn fetch(&self, doc_ids: &[String]) -> Result<Vec<Scored>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conds: Vec<Condition> = doc_ids
            .iter()
            .map(|id| Condition::matches("doc_id", id.clone()))
            .collect();
        let res = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.name)
                    .filter(Filter::should(conds))
                    .limit(FETCH_LIMIT)
                    .with_payload(true),
            )
            .await
            .with_context(|| format!("scrolling Qdrant collection `{}`", self.name))?;
        Ok(res
            .result
            .into_iter()
            .filter_map(|p| {
                chunk_from_payload(p.payload)
                    .ok()
                    .map(|chunk| Scored { chunk, score: 1.0 })
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
        assert_eq!(a, point_id("spell:x#0"));
        assert_ne!(a, point_id("spell:x#1"));
        assert!(uuid::Uuid::parse_str(&a).is_ok());
    }

    #[test]
    fn payload_roundtrips_through_chunk() {
        let c = chunk();
        let map: HashMap<String, Value> = payload_of(&c).unwrap().into();
        assert!(map.contains_key("trust_rank"), "rank exposed for filtering");
        let back = chunk_from_payload(map).unwrap();
        assert_eq!(back.chunk_id, c.chunk_id);
        assert_eq!(back.text, c.text);
        assert_eq!(back.trust, c.trust);
        assert_eq!(back.provenance.label, c.provenance.label);
    }

    #[test]
    fn fusion_parsing() {
        assert!(matches!(fusion_from_str("rrf").unwrap(), Fusion::Rrf));
        assert!(matches!(fusion_from_str("dbsf").unwrap(), Fusion::Dbsf));
        assert!(fusion_from_str("nope").is_err());
    }

    /// Live round-trips against a real Qdrant. Skipped unless `ENKI_QDRANT_TEST=1`
    /// (so CI, which has no server, is green). Uses a deterministic offline dense
    /// embedder + the zero-dep hashed sparse driver, and throwaway collections.
    #[tokio::test]
    async fn live_dense_and_hybrid() {
        if std::env::var("ENKI_QDRANT_TEST").as_deref() != Ok("1") {
            eprintln!("skipping: set ENKI_QDRANT_TEST=1 with a running Qdrant to run");
            return;
        }
        use crate::model::Metadata;
        use crate::sparse::HashedTfSparse;

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

        let url =
            std::env::var("ENKI_QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".into());
        let embedder: Arc<dyn Embedder> = Arc::new(Mock);
        let client = Arc::new(connect(&url, None).unwrap());

        let doc = |id: &str| Document {
            id: id.into(),
            content: "alpha beta gamma vecteur clairvoyance".into(),
            metadata: Metadata {
                label: format!("Doc {id}"),
                trust: TrustStatus::Canonical,
                ..Default::default()
            },
        };
        let query = |dense: Vec<f32>| EngineQuery {
            text: "alpha beta gamma".into(),
            dense: Some(dense),
            k: 5,
            filters: Default::default(),
        };
        let dense_vec = embedder
            .embed(&["alpha beta gamma".to_string()])
            .await
            .unwrap()
            .pop()
            .unwrap();

        // --- dense-only ---
        let name = "enki_test_dense";
        client.delete_collection(name).await.ok();
        let mut store = QdrantStore::open(&url, None, name, embedder.clone(), None).unwrap();
        store.upsert(vec![doc("d1")]).await.unwrap();
        let r = QdrantRetriever::dense(client.clone(), name);
        assert_eq!(
            r.retrieve(&query(dense_vec.clone())).await.unwrap()[0]
                .chunk
                .doc_id,
            "d1"
        );
        store.delete(&["d1".into()]).await.unwrap();
        assert!(
            r.retrieve(&query(dense_vec.clone()))
                .await
                .unwrap()
                .is_empty()
        );
        client.delete_collection(name).await.ok();

        // --- hybrid (dense + hashed sparse) ---
        let hname = "enki_test_hybrid";
        client.delete_collection(hname).await.ok();
        let sparse: Arc<dyn SparseEmbedder> = Arc::new(HashedTfSparse::new());
        let mut hstore =
            QdrantStore::open(&url, None, hname, embedder.clone(), Some(sparse.clone())).unwrap();
        hstore.upsert(vec![doc("h1")]).await.unwrap();
        let hr = QdrantRetriever::hybrid(client.clone(), hname, sparse, Fusion::Rrf);
        let hits = hr.retrieve(&query(dense_vec)).await.unwrap();
        assert!(
            !hits.is_empty(),
            "hybrid should retrieve the ingested chunk"
        );
        assert_eq!(hits[0].chunk.doc_id, "h1");
        client.delete_collection(hname).await.ok();
    }
}
