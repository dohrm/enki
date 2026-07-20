//! Sparse (lexical) embeddings — the second axis of a Qdrant hybrid collection.
//! A [`SparseEmbedder`] turns text into a [`SparseVector`] (parallel indices +
//! values); Qdrant fuses it with the dense vector server-side.
//!
//! Two drivers, both behind the trait (bring your own via [`crate::library::Library::builder`]):
//! - [`HashedTfSparse`] — zero-dep, offline, model-free: hashed term frequencies.
//!   Pair with Qdrant `Modifier::Idf` so the server applies IDF → BM25-ish. Runs
//!   anywhere (no ONNX), so it is the lightweight default.
//! - [`FastembedSparse`] (feature `fastembed`) — a learned sparse model
//!   (BGE-M3 sparse: multilingual, same family as the dense bge-m3; or SPLADE++).

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;

/// A sparse vector: parallel arrays of dimension indices and their weights,
/// sorted by index. The pivot type every driver produces.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Provider-agnostic sparse embedding seam (mirrors [`crate::embed::Embedder`]).
#[async_trait]
pub trait SparseEmbedder: Send + Sync {
    async fn embed_sparse(&self, texts: &[String]) -> Result<Vec<SparseVector>>;
}

// ---------- Driver 1: hashed term-frequency (zero-dep, offline) ----------

/// Model-free sparse encoder: tokenize → light multilingual stem → hash each term
/// to a `u32` dimension, value = normalized term frequency. **Meant to be paired
/// with Qdrant `Modifier::Idf`** (the server applies IDF; here we emit TF only),
/// which together approximate BM25. No ONNX, no download, deterministic.
///
/// Stopwords cover FR/DE/EN; stemming is intentionally conservative — only plural
/// normalization (see [`light_stem`]). It won't beat a learned model, but it's the
/// portable default when pulling onnxruntime isn't wanted.
#[derive(Debug, Clone, Default)]
pub struct HashedTfSparse;

impl HashedTfSparse {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl SparseEmbedder for HashedTfSparse {
    async fn embed_sparse(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
        Ok(texts.iter().map(|t| text_to_sparse(t)).collect())
    }
}

/// Encode one text as a normalized-TF sparse vector (indices sorted).
pub fn text_to_sparse(text: &str) -> SparseVector {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return SparseVector::default();
    }

    let mut tf: HashMap<u32, f32> = HashMap::new();
    for token in &tokens {
        *tf.entry(hash_token(token)).or_insert(0.0) += 1.0;
    }

    // Normalize by document length so long passages don't dominate the dot product.
    let total = tokens.len() as f32;
    let mut pairs: Vec<(u32, f32)> = tf.into_iter().map(|(i, c)| (i, c / total)).collect();
    pairs.sort_by_key(|(i, _)| *i); // deterministic + Qdrant expects sorted indices

    let (indices, values) = pairs.into_iter().unzip();
    SparseVector { indices, values }
}

/// Lowercase, split on non-alphanumerics, drop short tokens + stopwords, stem.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.chars().count() >= 2 && !is_stopword(s))
        .map(light_stem)
        .collect()
}

/// Conservative multilingual stemmer: **plural normalization only**. Linguistic
/// correctness is irrelevant — only that a query term and its document form hash
/// alike; `story`/`stories` must merge, unrelated terms must not. Greedy verb /
/// adverb / comparative rules (`-ing`, `-ed`, `-er`, `-ly`, `-est`, `-ment`) are
/// deliberately dropped: they cause false merges (`water→wat`, `master→mast`,
/// `comment→com`) that wreck lexical precision — and dense retrieval already
/// covers semantic recall.
///
/// Length gates are in **characters** (not bytes) so accented words behave the
/// same as ASCII ones; each rule strips an ASCII suffix, so byte-slicing stays on
/// a char boundary.
fn light_stem(word: &str) -> String {
    let chars = word.chars().count();
    let bytes = word.len();

    // -ies → -y : stories → story, companies → company
    if chars > 4 && word.ends_with("ies") {
        return format!("{}y", &word[..bytes - 3]);
    }
    // -eaux → -eau : châteaux → château (drop the trailing 'x')
    if chars > 5 && word.ends_with("eaux") {
        return word[..bytes - 1].to_string();
    }
    // -aux → -al : tribunaux → tribunal
    if chars > 4 && word.ends_with("aux") {
        return format!("{}al", &word[..bytes - 3]);
    }
    // -s plural, guarding non-plural endings (-ss stress, -us virus)
    if chars > 3 && word.ends_with('s') && !word.ends_with("ss") && !word.ends_with("us") {
        return word[..bytes - 1].to_string();
    }
    word.to_string()
}

/// FNV-1a 32-bit hash. Full `u32` space (no bucketing) → collisions negligible for
/// any realistic vocabulary; irreversible, but IDF (server-side) is index-based so
/// hashing is transparent to it.
fn hash_token(token: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in token.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// High-frequency function words (FR/DE/EN) that add noise without lexical value.
/// Language-scoped by design — a consumer on another language gets no filtering
/// (still correct, just noisier).
fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        // French
        "le" | "la" | "les" | "un" | "une" | "des" | "de" | "du" | "au" | "aux"
        | "et" | "ou" | "en" | "ce" | "se" | "ne" | "pas" | "que" | "qui" | "est"
        | "il" | "elle" | "nous" | "vous" | "ils" | "sont" | "dans" | "pour" | "par"
        | "sur" | "avec" | "son" | "sa" | "ses" | "cette" | "mais" | "plus"
        // German (excluding "an" — shared with English below)
        | "der" | "die" | "das" | "ein" | "eine" | "und" | "ist" | "ich" | "es"
        | "nicht" | "mit" | "auf" | "den" | "dem" | "von" | "zu" | "im" | "sich"
        | "sie" | "er" | "als" | "auch" | "aus" | "bei" | "nach" | "wie"
        | "hat" | "sind" | "wird" | "war" | "wenn" | "nur" | "noch"
        // English
        | "the" | "be" | "to" | "of" | "and" | "in" | "that" | "have" | "it"
        | "for" | "not" | "on" | "with" | "he" | "as" | "you" | "do" | "at"
        | "this" | "but" | "his" | "by" | "from" | "they" | "we" | "her"
        | "or" | "will" | "my" | "all" | "would" | "there" | "their" | "was"
        | "been" | "has" | "are" | "is" | "if" | "can" | "had" | "no" | "so"
    )
}

// ---------- Driver 2: learned sparse via fastembed (feature `fastembed`) ----------

#[cfg(feature = "fastembed")]
mod learned {
    use super::{SparseEmbedder, SparseVector};
    use anyhow::{Context, Result};
    use async_trait::async_trait;
    use fastembed::{SparseInitOptions, SparseModel, SparseTextEmbedding};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    /// Learned sparse embeddings via fastembed (ONNX). Default model **BGE-M3
    /// sparse** — multilingual (100+ languages), same family as the dense bge-m3.
    /// SPLADE++ (English) is also available via [`FastembedSparse::open`].
    pub struct FastembedSparse {
        model: Arc<Mutex<SparseTextEmbedding>>,
    }

    impl FastembedSparse {
        /// Load a sparse model. `cache_dir` reuses an existing fastembed/HF cache
        /// (e.g. a host app's model dir) instead of downloading; `None` = default.
        pub fn open(model: SparseModel, cache_dir: Option<PathBuf>) -> Result<Self> {
            let mut opts = SparseInitOptions::new(model.clone()).with_show_download_progress(true);
            if let Some(dir) = cache_dir {
                opts = opts.with_cache_dir(dir);
            }
            let model = SparseTextEmbedding::try_new(opts)
                .with_context(|| format!("loading sparse model {model}"))?;
            Ok(Self {
                model: Arc::new(Mutex::new(model)),
            })
        }
    }

    #[async_trait]
    impl SparseEmbedder for FastembedSparse {
        async fn embed_sparse(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let model = self.model.clone();
            let texts = texts.to_vec();
            // ONNX inference is blocking and `embed` needs `&mut` — off the async worker.
            let out = tokio::task::spawn_blocking(move || {
                let mut model = model
                    .lock()
                    .map_err(|_| anyhow::anyhow!("sparse model mutex poisoned"))?;
                model.embed(texts, None)
            })
            .await
            .context("sparse embed task")??;

            Ok(out
                .into_iter()
                .map(|e| SparseVector {
                    indices: e.indices.into_iter().map(|i| i as u32).collect(),
                    values: e.values,
                })
                .collect())
        }
    }
}

#[cfg(feature = "fastembed")]
pub use learned::FastembedSparse;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_filters_stopwords_multilingual() {
        let fr = tokenize("Le système de gestion est opérationnel");
        assert!(fr.contains(&"système".to_string()));
        assert!(fr.contains(&"gestion".to_string()));
        assert!(!fr.contains(&"le".to_string()) && !fr.contains(&"est".to_string()));

        let de = tokenize("Die Konfiguration ist nicht verfügbar");
        assert!(de.contains(&"konfiguration".to_string()) && de.contains(&"verfügbar".to_string()));
        assert!(!de.contains(&"die".to_string()) && !de.contains(&"nicht".to_string()));
    }

    #[test]
    fn stem_normalizes_plurals_only() {
        // Plurals merge (the win we keep).
        assert_eq!(light_stem("clans"), "clan");
        assert_eq!(light_stem("disciplines"), "discipline");
        assert_eq!(light_stem("stories"), "story");
        assert_eq!(light_stem("companies"), "company");
        assert_eq!(light_stem("tribunaux"), "tribunal");
        assert_eq!(light_stem("châteaux"), "château");
        // Preserved (no false stripping).
        assert_eq!(light_stem("stress"), "stress"); // -ss
        assert_eq!(light_stem("virus"), "virus"); // -us
        // Greedy rules dropped → no false merges.
        assert_eq!(light_stem("master"), "master"); // not "mast"
        assert_eq!(light_stem("running"), "running"); // not "runn"
        assert_eq!(light_stem("comment"), "comment"); // not "com"
    }

    #[test]
    fn stem_is_utf8_safe_on_accents() {
        // Length gates in chars, not bytes: a 3-char accented word ending in 's'.
        assert_eq!(light_stem("clés"), "clé"); // 4 chars, plural → "clé"
        assert_eq!(light_stem("île"), "île"); // no suffix rule fires, no panic
    }

    #[test]
    fn plural_and_singular_hash_alike() {
        let sing = text_to_sparse("clan vampire discipline");
        let plur = text_to_sparse("clans vampires disciplines");
        assert_eq!(sing.indices, plur.indices);
    }

    #[test]
    fn sparse_vector_is_sorted_normalized_and_deterministic() {
        let a = text_to_sparse("rust rust rust python systems");
        let b = text_to_sparse("rust rust rust python systems");
        assert_eq!(a, b, "deterministic");
        assert_eq!(a.indices.len(), a.values.len());
        for w in a.indices.windows(2) {
            assert!(w[0] < w[1], "indices sorted");
        }
        for &v in &a.values {
            assert!(v > 0.0 && v <= 1.0, "normalized TF in (0,1]");
        }
    }

    #[test]
    fn empty_and_stopword_only_yield_empty() {
        assert_eq!(text_to_sparse(""), SparseVector::default());
        assert!(text_to_sparse("the and is are to of").indices.is_empty());
    }

    #[tokio::test]
    async fn hashed_driver_maps_texts() {
        let d = HashedTfSparse::new();
        let out = d
            .embed_sparse(&["rust systems".into(), "".into()])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(!out[0].indices.is_empty());
        assert!(out[1].indices.is_empty());
    }
}
