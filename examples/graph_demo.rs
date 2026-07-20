//! Relation-graph demo (local backend, no Qdrant needed): the caller supplies
//! typed edges (as a campaign app like Quill would), and the agent traverses them
//! via the `neighbors` tool to answer a relational question.
//!
//!   ENKI_GRAPH=local cargo run --example graph_demo
//!
//! Uses whatever embedder/LLM the environment configures (see `.env.example`).
//! Ingestion + edges are written directly (like the CLI `index` command) because
//! `Library::open` expects the `corpus` collection to already exist.

use anyhow::Result;
use enki::config::Config;
use enki::embed::{Embedder, GenaiEmbedder};
use enki::graph::{Edge, GraphIndex, LocalGraph};
use enki::library::Library;
use enki::model::{Document, Metadata, TrustStatus};
use enki::store::{Index, LocalStore};
use enki::{providers, telemetry};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    telemetry::init_console();

    let mut cfg = Config::from_env();
    cfg.retrieval.cache_dir = ".cache-graph".into();
    cfg.retrieval.graph = "local".into(); // enable the graph tools

    let entity = |id: &str, name: &str, content: &str| Document {
        id: id.into(),
        content: content.into(),
        metadata: Metadata {
            label: name.into(),
            source: "campagne".into(),
            trust: TrustStatus::Canonical,
            ..Default::default()
        },
    };

    // 1. Ingest a few entities into `corpus` (content = the flesh).
    let embedder: Arc<dyn Embedder> = Arc::new(GenaiEmbedder::new(
        providers::client(
            &cfg.embed.provider,
            &cfg.embed.endpoint,
            cfg.embed.api_key.as_deref(),
        ),
        cfg.embed.model.clone(),
    ));
    let mut store = LocalStore::open(&cfg.retrieval.cache_dir, "corpus", embedder);
    store
        .upsert(vec![
            entity(
                "faction:ordre",
                "Faction: L'Ordre du Voile",
                "L'Ordre du Voile est une société secrète de mages qui gardent les failles entre les mondes.",
            ),
            entity(
                "pnj:lyra",
                "PNJ: Lyra Murmelune",
                "Lyra Murmelune est une ensorceleuse humaine, spécialiste des sorts de divination.",
            ),
            entity(
                "pnj:thorgrim",
                "PNJ: Thorgrim Barbe-de-Fer",
                "Thorgrim Barbe-de-Fer est un clerc nain, soigneur du groupe.",
            ),
        ])
        .await?;

    // 2. Supply the relations (the skeleton) — the caller owns these.
    let mut graph = LocalGraph::open(&cfg.retrieval.cache_dir);
    graph
        .upsert_edges(vec![
            Edge::new("pnj:lyra", "membre_de", "faction:ordre"),
            Edge::new("pnj:thorgrim", "membre_de", "faction:ordre"),
        ])
        .await?;
    println!("ingested 3 entities + 2 relations\n");

    // 3. Ask a relational question — the agent should find the faction, then call
    //    `neighbors` to enumerate its members.
    let library = Library::open(&cfg).await?;
    let question = "Qui sont les membres de l'Ordre du Voile ?";
    println!("> {question}\n");
    let answer = library.ask(question).await?;
    println!("{}\n", answer.markdown);
    for c in &answer.citations {
        if let Some(p) = &c.provenance {
            println!("  [{}] {}", c.number, p.label);
        }
    }
    Ok(())
}
