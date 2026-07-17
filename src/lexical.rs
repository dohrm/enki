//! Lexical (BM25) retrieval backed by [`tantivy`]. A `Modality::Lexical` source
//! that the engine fuses with the dense retriever via RRF — exact terms (spell
//! names, proper nouns) that dense embeddings blur are the point.
//!
//! The index is rebuilt from the full chunk set on every write (see
//! [`crate::store`]), so it stays in lock-step with the dense store. Each chunk's
//! text + label are indexed; the whole [`Chunk`] is stored as a JSON blob and
//! reconstructed on read, so this index is self-sufficient for lexical hits.

use crate::model::{Chunk, Scored};
use crate::retrieval::{Modality, Query, Retriever};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field, IndexRecordOption, STORED, Schema, TextFieldIndexing, TextOptions, Value,
};
use tantivy::tokenizer::{
    AsciiFoldingFilter, LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer,
};
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, doc};

/// Custom analyzer name: lowercase + ASCII-fold so `Dégâts` and `degats` match,
/// with exact terms preserved (no stemming — precision on names like "Boule de feu").
const TOKENIZER: &str = "enki_fr";

fn index_dir(cache_dir: &Path, name: &str) -> PathBuf {
    let stem = name.replace([':', '/'], "_");
    cache_dir.join(format!("{stem}.tantivy"))
}

fn analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(AsciiFoldingFilter)
        .build()
}

struct Fields {
    text: Field,
    label: Field,
    blob: Field,
}

fn build_schema() -> (Schema, Fields) {
    let indexing = TextFieldIndexing::default()
        .set_tokenizer(TOKENIZER)
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let indexed = TextOptions::default().set_indexing_options(indexing);

    let mut b = Schema::builder();
    let text = b.add_text_field("text", indexed.clone());
    let label = b.add_text_field("label", indexed);
    let blob = b.add_text_field("blob", STORED);
    (b.build(), Fields { text, label, blob })
}

/// Rebuild the lexical index for `name` from `chunks`. Wipes any existing index
/// first — a full rebuild is cheap (no embeddings) and keeps it exactly in sync.
pub fn build_index(cache_dir: &Path, name: &str, chunks: &[Chunk]) -> Result<()> {
    let dir = index_dir(cache_dir, name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).ok();
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let (schema, f) = build_schema();
    let index = Index::create_in_dir(&dir, schema)?;
    index.tokenizers().register(TOKENIZER, analyzer());

    let mut writer: IndexWriter = index.writer(50_000_000)?;
    for c in chunks {
        let blob = serde_json::to_string(c)?;
        writer.add_document(doc!(
            f.text => c.text.clone(),
            f.label => c.provenance.label.clone(),
            f.blob => blob,
        ))?;
    }
    writer.commit()?;
    Ok(())
}

/// Query-syntax has no place in a natural-language question: strip anything that
/// isn't alphanumeric or whitespace, leaving a bag of terms (default OR).
fn sanitize(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect()
}

/// BM25 retriever over a persisted [`build_index`] directory.
pub struct Tantivy {
    reader: tantivy::IndexReader,
    query_parser: QueryParser,
    blob: Field,
}

impl Tantivy {
    pub fn open(cache_dir: &Path, name: &str) -> Result<Self> {
        let dir = index_dir(cache_dir, name);
        anyhow::ensure!(
            dir.exists(),
            "lexical index `{name}` not built ({}) — run `index` with the `tantivy` feature",
            dir.display()
        );
        let index = Index::open_in_dir(&dir)?;
        index.tokenizers().register(TOKENIZER, analyzer());

        let schema = index.schema();
        let text = schema.get_field("text").context("missing text field")?;
        let label = schema.get_field("label").context("missing label field")?;
        let blob = schema.get_field("blob").context("missing blob field")?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        let query_parser = QueryParser::for_index(&index, vec![text, label]);
        Ok(Self {
            reader,
            query_parser,
            blob,
        })
    }
}

#[async_trait]
impl Retriever for Tantivy {
    fn modality(&self) -> Modality {
        Modality::Lexical
    }

    async fn retrieve(&self, q: &Query) -> Result<Vec<Scored>> {
        let searcher = self.reader.searcher();
        let Ok(query) = self.query_parser.parse_query(&sanitize(&q.text)) else {
            return Ok(Vec::new());
        };
        // Over-fetch a little so trust filtering doesn't drop us below k.
        let top = searcher.search(&query, &TopDocs::with_limit(q.k * 2))?;

        let mut out = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let Some(blob) = doc.get_first(self.blob).and_then(|v| v.as_str()) else {
                continue;
            };
            let chunk: Chunk = serde_json::from_str(blob).context("decoding stored chunk")?;
            if q.filters.allows(&chunk) {
                out.push(Scored { chunk, score });
            }
        }
        out.truncate(q.k);
        Ok(out)
    }
}
