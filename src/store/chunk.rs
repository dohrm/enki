//! Library-side chunking: a [`Document`] becomes one or more [`Chunk`]s.

use crate::model::{Chunk, Document, Provenance};

/// Max characters per chunk before splitting. ~1k tokens: precise for retrieval,
/// well within embedding-model limits.
pub(crate) const MAX_CHUNK_CHARS: usize = 4000;

pub(crate) fn chunk_document(doc: &Document) -> Vec<Chunk> {
    let windows = split_on_paragraphs(doc.content.trim(), MAX_CHUNK_CHARS);
    windows
        .into_iter()
        .enumerate()
        .map(|(part, text)| Chunk {
            chunk_id: format!("{}#{}", doc.id, part),
            doc_id: doc.id.clone(),
            text,
            tier: doc.metadata.tier,
            trust: doc.metadata.trust,
            tags: doc.metadata.tags.clone(),
            provenance: Provenance {
                source: doc.metadata.source.clone(),
                label: doc.metadata.label.clone(),
                page_range: doc.metadata.page_range,
                extra: doc.metadata.extra.clone(),
            },
        })
        .collect()
}

/// Split on blank lines so a window never cuts mid-paragraph (unless a single
/// paragraph exceeds the budget).
pub(crate) fn split_on_paragraphs(text: &str, max_chars: usize) -> Vec<String> {
    let mut windows = Vec::new();
    let mut current = String::new();
    for para in text.split("\n\n") {
        if !current.is_empty() && current.chars().count() + para.chars().count() > max_chars {
            windows.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if !current.trim().is_empty() {
        windows.push(current);
    }
    if windows.is_empty() && !text.is_empty() {
        windows.push(text.to_string());
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Document, Metadata};

    #[test]
    fn keeps_small_text_as_one_window() {
        let w = split_on_paragraphs("one\n\ntwo\n\nthree", 4000);
        assert_eq!(w, vec!["one\n\ntwo\n\nthree"]);
    }

    #[test]
    fn splits_on_paragraph_boundary_not_mid_paragraph() {
        // Two ~30-char paragraphs, budget 40 → each paragraph its own window.
        let a = "a".repeat(30);
        let b = "b".repeat(30);
        let text = format!("{a}\n\n{b}");
        let w = split_on_paragraphs(&text, 40);
        assert_eq!(w, vec![a, b]);
    }

    #[test]
    fn oversized_single_paragraph_is_kept_whole() {
        let big = "x".repeat(100);
        let w = split_on_paragraphs(&big, 40);
        assert_eq!(w, vec![big]); // never lose content, even over budget
    }

    #[test]
    fn chunk_ids_are_doc_id_indexed_and_carry_metadata() {
        let a = "a".repeat(30);
        let b = "b".repeat(30);
        let doc = Document {
            id: "doc1".to_string(),
            content: format!("{a}\n\n{b}"),
            metadata: Metadata {
                tier: 2,
                label: "Lieu: Taverne".to_string(),
                ..Default::default()
            },
        };
        let chunks = chunk_document(&doc);
        assert_eq!(chunks.len(), 1); // fits one window at default budget
        assert_eq!(chunks[0].chunk_id, "doc1#0");
        assert_eq!(chunks[0].doc_id, "doc1");
        assert_eq!(chunks[0].tier, 2);
        assert_eq!(chunks[0].provenance.label, "Lieu: Taverne");
    }
}
