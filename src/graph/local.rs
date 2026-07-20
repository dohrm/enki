//! Zero-dep on-disk graph: one global `graph.json` (a list of [`Edge`]s) loaded
//! into forward + reverse adjacency. Enough for bounded neighbour lookups over a
//! campaign-sized graph — not a graph database, by design.

use super::{Direction, Edge, GraphIndex, GraphStore, Neighbor};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

type Adjacency = HashMap<String, Vec<(String, String)>>; // node -> [(predicate, other)]

pub struct LocalGraph {
    path: PathBuf,
    edges: Vec<Edge>,
    fwd: Adjacency, // from -> [(predicate, to)]
    rev: Adjacency, // to   -> [(predicate, from)]
}

impl LocalGraph {
    /// Open (or start empty) the global graph under `cache_dir`.
    pub fn open(cache_dir: &Path) -> Self {
        let path = cache_dir.join("graph.json");
        let edges: Vec<Edge> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let (fwd, rev) = build(&edges);
        Self {
            path,
            edges,
            fwd,
            rev,
        }
    }

    fn rebuild(&mut self) {
        let (fwd, rev) = build(&self.edges);
        self.fwd = fwd;
        self.rev = rev;
    }

    fn persist(&self) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(&self.path, serde_json::to_string(&self.edges)?)
            .with_context(|| format!("writing {}", self.path.display()))
    }

    /// Number of edges (for tests / diagnostics).
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

fn build(edges: &[Edge]) -> (Adjacency, Adjacency) {
    let mut fwd: Adjacency = HashMap::new();
    let mut rev: Adjacency = HashMap::new();
    for e in edges {
        fwd.entry(e.from.clone())
            .or_default()
            .push((e.predicate.clone(), e.to.clone()));
        rev.entry(e.to.clone())
            .or_default()
            .push((e.predicate.clone(), e.from.clone()));
    }
    (fwd, rev)
}

#[async_trait]
impl GraphStore for LocalGraph {
    async fn neighbors(
        &self,
        id: &str,
        predicate: Option<&str>,
        dir: Direction,
        limit: usize,
    ) -> Result<Vec<Neighbor>> {
        let want = |p: &str| predicate.is_none_or(|w| w == p);
        let mut out = Vec::new();
        if matches!(dir, Direction::Out | Direction::Both) {
            for (p, to) in self.fwd.get(id).into_iter().flatten() {
                if want(p) {
                    out.push(Neighbor {
                        id: to.clone(),
                        predicate: p.clone(),
                        direction: Direction::Out,
                    });
                }
            }
        }
        if matches!(dir, Direction::In | Direction::Both) {
            for (p, from) in self.rev.get(id).into_iter().flatten() {
                if want(p) {
                    out.push(Neighbor {
                        id: from.clone(),
                        predicate: p.clone(),
                        direction: Direction::In,
                    });
                }
            }
        }
        out.truncate(limit);
        Ok(out)
    }
}

#[async_trait]
impl GraphIndex for LocalGraph {
    async fn upsert_edges(&mut self, edges: Vec<Edge>) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let existing: HashSet<Edge> = self.edges.iter().cloned().collect();
        for e in edges {
            if !existing.contains(&e) {
                self.edges.push(e);
            }
        }
        self.rebuild();
        self.persist()
    }

    async fn delete_nodes(&mut self, ids: &[String]) -> Result<()> {
        let drop: HashSet<&str> = ids.iter().map(String::as_str).collect();
        let before = self.edges.len();
        self.edges
            .retain(|e| !drop.contains(e.from.as_str()) && !drop.contains(e.to.as_str()));
        if self.edges.len() == before {
            return Ok(());
        }
        self.rebuild();
        self.persist()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn upsert_neighbors_directions_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let mut g = LocalGraph::open(dir.path());
        g.upsert_edges(vec![
            Edge::new("pnj:lyra", "membre_de", "faction:ordre"),
            Edge::new("pnj:thorgrim", "membre_de", "faction:ordre"),
            Edge::new("pnj:lyra", "possede", "objet:lame"),
        ])
        .await
        .unwrap();

        // Out: what lyra points to.
        let out = g
            .neighbors("pnj:lyra", None, Direction::Out, 10)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        // Predicate filter.
        let membership = g
            .neighbors("pnj:lyra", Some("membre_de"), Direction::Out, 10)
            .await
            .unwrap();
        assert_eq!(membership.len(), 1);
        assert_eq!(membership[0].id, "faction:ordre");
        // In: who points at the faction (both members).
        let members = g
            .neighbors("faction:ordre", None, Direction::In, 10)
            .await
            .unwrap();
        assert_eq!(members.len(), 2);

        // Idempotent upsert.
        g.upsert_edges(vec![Edge::new("pnj:lyra", "membre_de", "faction:ordre")])
            .await
            .unwrap();
        assert_eq!(g.edge_count(), 3);

        // Delete a node drops its edges (both directions).
        g.delete_nodes(&["pnj:lyra".to_string()]).await.unwrap();
        assert_eq!(g.edge_count(), 1);
        assert!(
            g.neighbors("pnj:lyra", None, Direction::Both, 10)
                .await
                .unwrap()
                .is_empty()
        );

        // Persisted: a fresh handle sees the same state.
        let g2 = LocalGraph::open(dir.path());
        assert_eq!(g2.edge_count(), 1);
    }
}
