//! Registre de passages, scoped conversation : bijection `chunk_id ↔ handle`.
//!
//! Le LLM ne voit que des handles courts (`s7`), jamais un chunk_id/uuid.
//! Accumulé, jamais renuméroté, dédupliqué par chunk_id (même chunk ressorti par
//! deux `search` → même handle).

use crate::model::Chunk;
use std::collections::HashMap;

#[derive(Default)]
pub struct Registry {
    handle_by_chunk: HashMap<String, String>,
    chunk_by_handle: HashMap<String, Chunk>,
    next: usize,
}

impl Registry {
    /// Interne un chunk et renvoie son handle (stable pour la conversation).
    pub fn intern(&mut self, chunk: &Chunk) -> String {
        if let Some(h) = self.handle_by_chunk.get(&chunk.chunk_id) {
            return h.clone();
        }
        self.next += 1;
        let handle = format!("s{}", self.next);
        self.handle_by_chunk
            .insert(chunk.chunk_id.clone(), handle.clone());
        self.chunk_by_handle.insert(handle.clone(), chunk.clone());
        handle
    }

    /// Résout un handle émis par le LLM vers son chunk (→ provenance, vérif quote).
    pub fn get(&self, handle: &str) -> Option<&Chunk> {
        self.chunk_by_handle.get(handle)
    }

    /// Nombre de chunks distincts internés (sert à détecter les recherches
    /// stériles : une recherche qui n'ajoute rien ne fait pas croître ce compteur).
    pub fn len(&self) -> usize {
        self.chunk_by_handle.len()
    }

    /// Vrai si aucun chunk n'a encore été interné.
    pub fn is_empty(&self) -> bool {
        self.chunk_by_handle.is_empty()
    }
}
