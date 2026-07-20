//! Relation graph — the corpus-wide *skeleton* (who relates to whom), joined to
//! the vector store's *content* by entity id (`doc_id` / `semantic_id`).
//!
//! Enki does not *infer* the graph (no LLM extraction, unlike graph-RAG systems):
//! the caller **supplies** typed edges via [`crate::library::Library::relate`]
//! (a campaign app already has them). One global graph, transverse to collections
//! and tiers. The library only exposes bounded neighbour lookups — traversal /
//! multi-hop reasoning stays in the agentic loop, not in a query language here.
//!
//! Split: [`local`] (a zero-dep on-disk adjacency, the desktop profile). A server
//! backend (Neo4j) can slot behind the same traits later.

mod local;

pub use local::LocalGraph;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A directed, typed relation between two entities (by id).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub predicate: String,
    pub to: String,
}

impl Edge {
    pub fn new(
        from: impl Into<String>,
        predicate: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        Self {
            from: from.into(),
            predicate: predicate.into(),
            to: to.into(),
        }
    }
}

/// Which way to walk an edge relative to the queried node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `node --predicate--> neighbour`
    Out,
    /// `neighbour --predicate--> node`
    In,
    Both,
}

impl Direction {
    pub fn as_str(self) -> &'static str {
        match self {
            Direction::Out => "out",
            Direction::In => "in",
            Direction::Both => "both",
        }
    }
}

/// A node reachable from the queried one, with the relation that links them.
#[derive(Debug, Clone)]
pub struct Neighbor {
    pub id: String,
    pub predicate: String,
    pub direction: Direction,
}

/// Read seam: bounded neighbour lookup. Behind `Arc` for the agentic graph tool.
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Neighbours of `id`, optionally filtered to one `predicate`, in `dir`,
    /// capped at `limit`. Empty if the node is unknown.
    async fn neighbors(
        &self,
        id: &str,
        predicate: Option<&str>,
        dir: Direction,
        limit: usize,
    ) -> Result<Vec<Neighbor>>;
}

/// Write seam (the counterpart of [`crate::store::Index`] for the graph).
#[async_trait]
pub trait GraphIndex: Send + Sync {
    /// Add edges (idempotent — duplicates are ignored).
    async fn upsert_edges(&mut self, edges: Vec<Edge>) -> Result<()>;
    /// Remove nodes and every edge touching them (either endpoint).
    async fn delete_nodes(&mut self, ids: &[String]) -> Result<()>;
}
