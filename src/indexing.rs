//! Manifest importer — one *producer* for the store's [`crate::store::Index`]
//! write interface. Parses the document manifest (hierarchical outline) into
//! [`Document`]s; the store does the chunking and embedding. The living "instance"
//! tier is fed by the calling application through the same interface instead.

use crate::model::{Document, Metadata};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Section {
    section_ref: String,
    title: String,
    level: u32,
    page_start: u32,
    page_end: u32,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    document_name: String,
    sections: Vec<Section>,
}

/// Parse the manifest into documents. Each section becomes one document; its
/// hierarchy path is the citation label AND the contextual-embedding prefix
/// (e.g. "GANGREL > Disciplines" disambiguates otherwise-identical sections).
pub fn manifest_documents(manifest_path: &Path) -> Result<Vec<Document>> {
    let raw = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("reading manifest {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&raw).context("parsing manifest")?;
    let source = if manifest.document_name.is_empty() {
        "document".to_string()
    } else {
        manifest.document_name
    };

    let mut docs = Vec::new();
    // Ancestor titles by depth: `path[i]` is the title at level `i`. Rebuilt at
    // each section so the label is the full breadcrumb (the manifest nests up to
    // 5 levels), e.g. "Comment jouer > Tests d20 > Jets de sauvegarde".
    let mut path: Vec<String> = Vec::new();

    for section in &manifest.sections {
        let lvl = section.level as usize;
        path.truncate(lvl);
        while path.len() < lvl {
            path.push(String::new()); // pad if a level is skipped
        }
        // Trim table-of-contents dot leaders (e.g. "Sorts........" → "Sorts").
        let title = section
            .title
            .trim_end_matches(|c: char| c == '.' || c.is_whitespace())
            .to_string();
        path.push(title);

        if section.content.trim().is_empty() {
            continue;
        }

        let label = path
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" > ");

        docs.push(Document {
            id: section.section_ref.clone(),
            content: section.content.clone(),
            metadata: Metadata {
                tier: 0,
                trust: crate::model::TrustStatus::Canonical,
                source: source.clone(),
                label,
                tags: Vec::new(),
                page_range: Some((section.page_start, section.page_end)),
                extra: Default::default(),
            },
        });
    }

    Ok(docs)
}
