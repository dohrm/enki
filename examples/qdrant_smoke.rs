//! Live smoke test of the Qdrant backend: ingest a couple of documents straight
//! into Qdrant, then open the library over it and query / ask.
//!
//!   ENKI_QDRANT_URL=http://localhost:6334 \
//!   cargo run --example qdrant_smoke --features qdrant
//!
//! Uses whatever embedder/LLM the environment configures (Ollama by default, see
//! `.env.example`). Ingestion goes through `QdrantStore` directly — the same way
//! the CLI's `index` command uses `LocalStore` — because `Library::open` expects
//! the `corpus` collection to already exist.

#[cfg(not(feature = "qdrant"))]
fn main() {
    eprintln!(
        "this example needs the `qdrant` feature: cargo run --example qdrant_smoke --features qdrant"
    );
}

#[cfg(feature = "qdrant")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use enki::config::Config;
    use enki::embed::{Embedder, GenaiEmbedder};
    use enki::library::Library;
    use enki::model::{Document, Metadata, TrustStatus};
    use enki::providers;
    use enki::qdrant::QdrantStore;
    use enki::store::Index;
    use std::sync::Arc;

    dotenvy::dotenv().ok();
    enki::telemetry::init_console();

    let mut cfg = Config::from_env();
    cfg.retrieval.backend = "qdrant".into(); // force the backend for the demo

    let doc = |id: &str, label: &str, content: &str| Document {
        id: id.into(),
        content: content.into(),
        metadata: Metadata {
            label: label.into(),
            source: "SRD".into(),
            trust: TrustStatus::Canonical,
            ..Default::default()
        },
    };

    // 1. Ingest a tiny corpus directly into the Qdrant `corpus` collection.
    let embedder: Arc<dyn Embedder> = Arc::new(GenaiEmbedder::new(
        providers::client(
            &cfg.embed.provider,
            &cfg.embed.endpoint,
            cfg.embed.api_key.as_deref(),
        ),
        cfg.embed.model.clone(),
    ));
    let mut store = QdrantStore::open(
        &cfg.retrieval.qdrant_url,
        cfg.retrieval.qdrant_api_key.as_deref(),
        "corpus",
        embedder,
    )?;
    store
        .upsert(vec![
            doc(
                "note:boule",
                "Sorts > Boule de feu",
                "La Boule de feu inflige 8d6 dégâts de feu à toutes les créatures dans un rayon de 6 mètres (jet de sauvegarde de Dextérité pour la moitié).",
            ),
            doc(
                "note:soin",
                "Sorts > Soins",
                "Le sort Soins restaure un nombre de points de vie égal à 1d8 + le modificateur de caractéristique d'incantation.",
            ),
        ])
        .await?;
    println!(
        "ingested 2 documents into Qdrant `corpus` ({})",
        cfg.retrieval.qdrant_url
    );

    // 2. Open the library over Qdrant and use the normal read API.
    let library = Library::open(&cfg).await?;
    println!(
        "opened library: {} collection(s)\n",
        library.collection_count()
    );

    println!("-- search --");
    for h in library.search("dégâts de la boule de feu", 3).await? {
        println!(
            "  [{:.3}] {} — {}",
            h.score, h.chunk.provenance.label, h.chunk.text
        );
    }

    println!("\n-- ask --");
    let answer = library
        .ask("Combien de dégâts inflige la Boule de feu ?")
        .await?;
    println!("{}", answer.markdown);
    Ok(())
}
