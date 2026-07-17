//! Core types, shared by the write side (ingestion), the store, and retrieval.
//! Domain-agnostic: the library knows "documents", "chunks", "tiers" — never what
//! the corpus is about (that context is injected).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Source authority rank. Higher = more authoritative. Priority between tiers is
/// applied at search time, never frozen in the index. Labels come from config.
pub type Tier = u8;

/// Trust state of a source. A search-time filter. `Canonical` = a reference source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TrustStatus {
    #[default]
    Draft,
    PublicNote,
    Endorsed,
    Canonical,
}

impl TrustStatus {
    /// Ordinal used for `trust_min` filtering (higher = more trusted).
    pub fn rank(self) -> u8 {
        match self {
            TrustStatus::Draft => 0,
            TrustStatus::PublicNote => 1,
            TrustStatus::Endorsed => 2,
            TrustStatus::Canonical => 3,
        }
    }
}

/// Provenance resolved from a handle at render time. Unified across source types:
/// a display `label` (breadcrumb) + an optional page locator + free-form `extra`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    /// Citation breadcrumb, e.g. "LE SANG > Pouvoirs du Sang" or "PNJ: Carmilla".
    pub label: String,
    /// Page span for paginated sources; `None` otherwise.
    pub page_range: Option<(u32, u32)>,
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

/// Indexed and retrieved unit. Produced by chunking a [`Document`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    /// Id of the source document (chunks of the same document share it; used to
    /// replace a document's chunks on upsert).
    pub doc_id: String,
    pub text: String,
    pub tier: Tier,
    pub trust: TrustStatus,
    #[serde(default)]
    pub tags: Vec<String>,
    pub provenance: Provenance,
}

impl Chunk {
    /// Text fed to the embedder: the provenance label is prepended so titles
    /// (which carry disambiguating info) shape the vector — contextual retrieval.
    pub fn embed_input(&self) -> String {
        if self.provenance.label.is_empty() {
            self.text.clone()
        } else {
            format!("{}\n\n{}", self.provenance.label, self.text)
        }
    }
}

/// Caller-facing metadata attached to an ingested [`Document`]. Drives both
/// provenance (citation display) and filters (tier / trust / tags).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub tier: Tier,
    pub trust: TrustStatus,
    pub source: String,
    pub label: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub page_range: Option<(u32, u32)>,
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

/// A unit of content pushed through the ingestion interface. The library chunks
/// and embeds it. `id` is caller-owned and makes upsert idempotent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: Metadata,
}

/// Chunk + relevance score for a given query.
#[derive(Debug, Clone)]
pub struct Scored {
    pub chunk: Chunk,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(label: &str, text: &str) -> Chunk {
        Chunk {
            chunk_id: "c#0".into(),
            doc_id: "c".into(),
            text: text.into(),
            tier: 0,
            trust: TrustStatus::Draft,
            tags: vec![],
            provenance: Provenance {
                label: label.into(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn embed_input_prepends_label_as_context() {
        let c = chunk("GANGREL > Disciplines", "Animalité, Force d'âme…");
        assert_eq!(
            c.embed_input(),
            "GANGREL > Disciplines\n\nAnimalité, Force d'âme…"
        );
    }

    #[test]
    fn embed_input_without_label_is_just_text() {
        let c = chunk("", "raw text");
        assert_eq!(c.embed_input(), "raw text");
    }

    #[test]
    fn trust_rank_is_ordered() {
        assert!(TrustStatus::Canonical.rank() > TrustStatus::Endorsed.rank());
        assert!(TrustStatus::Endorsed.rank() > TrustStatus::PublicNote.rank());
        assert!(TrustStatus::PublicNote.rank() > TrustStatus::Draft.rank());
        assert_eq!(TrustStatus::default(), TrustStatus::Draft);
    }
}
