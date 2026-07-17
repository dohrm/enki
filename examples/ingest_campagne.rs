//! Demo producer: ingest an Obsidian-style campaign vault into the `instance`
//! collection through the library's `Index::upsert` interface. This lives OUTSIDE
//! the library on purpose — a real caller (e.g. a Tauri app) owns its source and
//! feeds enki the same way. The library stays domain-agnostic.
//!
//!   cargo run --example ingest_campagne [-- <vault_dir>]   # default: data/Campagne

use anyhow::{Context, Result};
use enki::config::Config;
use enki::embed::{Embedder, GenaiEmbedder};
use enki::model::{Document, Metadata, TrustStatus};
use enki::providers;
use enki::store::{Index, LocalStore};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Deserialize, Default)]
struct FrontMatter {
    semantic_id: Option<String>,
    name: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    /// Visibility (gm_only / players / public). Carried as metadata for now; a
    /// first-class `Filters.scope` visibility gate is a planned enki feature.
    scope: Option<String>,
    status: Option<String>,
}

/// Directory name → human category for the citation label.
fn category_label(cat: &str) -> &str {
    match cat {
        "pnj" => "PNJ",
        "pj" => "PJ",
        "lieux" | "locations" => "Lieu",
        "intrigues" | "quetes" | "quests" => "Intrigue",
        "factions" => "Faction",
        "notes" => "Note",
        "objets" | "objects" => "Objet",
        "regles" | "rules" => "Règle",
        _ => "Doc",
    }
}

fn split_frontmatter(raw: &str) -> (&str, &str) {
    if let Some(rest) = raw.strip_prefix("---\n")
        && let Some(end) = rest.find("\n---\n")
    {
        return (&rest[..end], &rest[end + 5..]);
    }
    ("", raw)
}

/// Split a markdown body into (section_title, body) on `## ` headers. Content
/// before the first `##` is an intro section with an empty title. Header-aware
/// chunking isolates focused subsections (e.g. "Disciplines") and lets the label
/// path disambiguate — the same win as the corpus title path.
fn split_sections(body: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let mut title = String::new();
    let mut buf = String::new();
    for line in body.lines() {
        if let Some(h) = line.strip_prefix("## ") {
            if !buf.trim().is_empty() {
                sections.push((std::mem::take(&mut title), std::mem::take(&mut buf)));
            }
            title = h.trim().to_string();
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    if !buf.trim().is_empty() {
        sections.push((title, buf));
    }
    sections
}

fn collect_md(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_md(&path, out);
        } else if path.extension().is_some_and(|e| e == "md") {
            out.push(path);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    enki::telemetry::init_console();
    let cfg = Config::from_env();
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/Campagne".to_string())
        .into();

    let mut files = Vec::new();
    collect_md(&dir, &mut files);
    files.sort();

    let mut docs = Vec::new();
    for path in &files {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let (fm_raw, body) = split_frontmatter(&raw);
        if body.trim().is_empty() {
            continue;
        }
        let fm: FrontMatter = serde_yaml::from_str(fm_raw).unwrap_or_default();

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("doc")
            .to_string();
        let category = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("doc");
        let name = fm.name.clone().unwrap_or_else(|| stem.clone());
        let base_id = fm.semantic_id.clone().unwrap_or(stem);
        let cat = category_label(category);

        let mut extra = BTreeMap::new();
        if let Some(k) = &fm.kind {
            extra.insert("type".to_string(), k.clone());
        }
        if let Some(s) = &fm.status {
            extra.insert("status".to_string(), s.clone());
        }
        if let Some(sc) = &fm.scope {
            extra.insert("scope".to_string(), sc.clone());
        }
        if !fm.aliases.is_empty() {
            extra.insert("aliases".to_string(), fm.aliases.join(", "));
        }

        // Carry the visibility scope as a tag too, so it survives into retrieval
        // metadata until a first-class visibility filter exists.
        let mut tags = fm.tags.clone();
        if let Some(sc) = &fm.scope {
            tags.push(format!("scope:{sc}"));
        }

        // One document per `##` section → focused, well-labelled chunks.
        for (i, (section, sec_body)) in split_sections(body).into_iter().enumerate() {
            let label = if section.is_empty() {
                format!("{cat}: {name}")
            } else {
                format!("{cat}: {name} > {section}")
            };
            docs.push(Document {
                id: format!("{base_id}#{i}"),
                content: sec_body,
                metadata: Metadata {
                    tier: 1, // instance — more specific than the book (tier 0)
                    trust: TrustStatus::Endorsed,
                    source: "campagne".to_string(),
                    label,
                    tags: tags.clone(),
                    page_range: None,
                    extra: extra.clone(),
                },
            });
        }
    }

    println!(
        "campaign documents: {} (from {})",
        docs.len(),
        dir.display()
    );

    let embedder: Arc<dyn Embedder> = Arc::new(GenaiEmbedder::new(
        providers::client(
            &cfg.embed.provider,
            &cfg.embed.endpoint,
            cfg.embed.api_key.as_deref(),
        ),
        cfg.embed.model.clone(),
    ));
    let mut store = LocalStore::open(&cfg.retrieval.cache_dir, "instance", embedder);
    store.upsert(docs).await?;
    println!("instance collection ready");
    Ok(())
}
