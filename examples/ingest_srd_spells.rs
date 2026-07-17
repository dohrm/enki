//! Producer: one Document per SRD spell, into the `spells` collection.
//!
//! The manifest's "Description des sorts" is a single ~340 KB section; size-only
//! chunking lumps ~10 spells per chunk, so a query like "Boule de feu damage"
//! matches a blurry multi-spell wall and the exact `8d6` gets buried. This
//! producer splits that section on spell boundaries — one focused, labelled doc
//! per spell ("Sorts > Boule de feu"), with the spell's classes as tags (enabling
//! "unusual for this class" questions) and a per-spell page for citations.
//!
//! Lives outside the library on purpose (domain-specific parsing) — same contract
//! as `ingest_campagne`.
//!
//!   ENKI_MANIFEST=…/document.manifest.json ENKI_CACHE_DIR=.cache-dnd \
//!     cargo run --example ingest_srd_spells

use anyhow::{Context, Result};
use enki::config::Config;
use enki::embed::{Embedder, GenaiEmbedder};
use enki::model::{Document, Metadata, TrustStatus};
use enki::providers;
use enki::store::{Index, LocalStore};
use regex::Regex;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
struct Section {
    title: String,
    #[serde(default)]
    content: String,
}
#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    document_name: String,
    sections: Vec<Section>,
}

struct Spell {
    name: String,
    school: String,
    level: u32, // 0 = cantrip
    classes: Vec<String>,
    page: u32,
    text: String,
}

fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Parse the "Description des sorts" blob into individual spells. A spell header
/// is `<Name> <School> du <N>e niveau (<classes>)` or `<Name> <School> mineure
/// (<classes>)` (cantrip), with spells separated by blank lines.
fn parse_spells(content: &str, default_page: u32) -> Vec<Spell> {
    let schools = "Abjuration|Divination|Enchantement|Évocation|Illusion|Invocation|Nécromancie|Transmutation";
    let header = Regex::new(&format!(
        r"({schools})\s+(?:du\s+(\d+)\S*\s+niveau|mineure)\s*\(([^)]+)\)"
    ))
    .unwrap();
    let page_marker = Regex::new(r"Document de Référence du Système 5\.2\.1\s+(\d+)").unwrap();

    // Page markers (position → page number), to attach a page to each spell.
    let pages: Vec<(usize, u32)> = page_marker
        .captures_iter(content)
        .filter_map(|c| Some((c.get(0)?.start(), c[1].parse().ok()?)))
        .collect();
    let page_at = |pos: usize| {
        pages
            .iter()
            .rev()
            .find(|(p, _)| *p <= pos)
            .map(|(_, n)| *n)
            .unwrap_or(default_page)
    };

    // Each spell starts at the blank line before its name; the header match gives
    // school/level/classes and the name is the text between that blank line and
    // the school keyword.
    let starts: Vec<(usize, regex::Captures)> = header
        .captures_iter(content)
        .map(|c| {
            let school_at = c.get(1).unwrap().start();
            let name_start = content[..school_at]
                .rfind("\n\n")
                .map(|i| i + 2)
                .unwrap_or(0);
            (name_start, c)
        })
        .collect();

    let mut spells = Vec::with_capacity(starts.len());
    for i in 0..starts.len() {
        let (name_start, caps) = &starts[i];
        let school_at = caps.get(1).unwrap().start();
        let end = starts.get(i + 1).map(|(s, _)| *s).unwrap_or(content.len());

        let name = content[*name_start..school_at].trim();
        if name.is_empty() || name.len() > 60 {
            continue; // guard against a stray in-body match
        }
        let raw = &content[*name_start..end];
        let text = page_marker
            .replace_all(raw, " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        spells.push(Spell {
            name: name.to_string(),
            school: caps.get(1).unwrap().as_str().to_string(),
            level: caps
                .get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0),
            classes: caps[3]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            page: page_at(*name_start),
            text,
        });
    }
    spells
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    enki::telemetry::init_console();
    let cfg = Config::from_env();

    let raw = std::fs::read_to_string(&cfg.indexing.manifest_path)
        .with_context(|| format!("reading {}", cfg.indexing.manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&raw).context("parsing manifest")?;
    let source = if manifest.document_name.is_empty() {
        "SRD".to_string()
    } else {
        manifest.document_name.clone()
    };

    let section = manifest
        .sections
        .iter()
        .find(|s| s.title.starts_with("Description des sorts"))
        .context("no 'Description des sorts' section in manifest")?;

    let spells = parse_spells(&section.content, 113);
    println!("parsed {} spells", spells.len());

    let docs: Vec<Document> = spells
        .into_iter()
        .map(|s| {
            let mut tags = vec![
                s.school.to_lowercase(),
                if s.level == 0 {
                    "sort-mineur".to_string()
                } else {
                    format!("niveau-{}", s.level)
                },
            ];
            tags.extend(s.classes.iter().map(|c| c.to_lowercase()));
            Document {
                id: format!("spell:{}", slug(&s.name)),
                content: s.text,
                metadata: Metadata {
                    tier: 0, // book authority, same as corpus
                    trust: TrustStatus::Canonical,
                    source: source.clone(),
                    label: format!("Sorts > {}", s.name),
                    tags,
                    page_range: Some((s.page, s.page)),
                    extra: Default::default(),
                },
            }
        })
        .collect();

    let embedder: Arc<dyn Embedder> = Arc::new(GenaiEmbedder::new(
        providers::client(
            &cfg.embed.provider,
            &cfg.embed.endpoint,
            cfg.embed.api_key.as_deref(),
        ),
        cfg.embed.model.clone(),
    ));
    let mut store = LocalStore::open(&cfg.retrieval.cache_dir, "spells", embedder);
    store.upsert(docs).await?;
    println!("spells collection ready");
    Ok(())
}
