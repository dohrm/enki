//! Read side: a persisted collection loaded for retrieval. Vectors live in RAM by
//! default, or mmap (feature `fs`) for fast cold start / low RAM.

use super::{load_docs, paths};
use crate::model::Chunk;
use anyhow::{Context, Result};
use std::path::Path;

enum VecData {
    #[cfg(not(feature = "fs"))]
    Ram(Vec<f32>),
    #[cfg(feature = "fs")]
    Mapped(memmap2::Mmap),
}

impl VecData {
    fn as_slice(&self) -> &[f32] {
        match self {
            #[cfg(not(feature = "fs"))]
            VecData::Ram(v) => v,
            // mmap base is page-aligned → offset 0 is 4-aligned → zero-copy cast is sound.
            #[cfg(feature = "fs")]
            VecData::Mapped(m) => bytemuck::cast_slice(&m[..]),
        }
    }
}

/// Read handle over a persisted collection: chunks + row-aligned vectors.
/// RAM by default; mmap-backed under feature `fs`.
pub struct VectorStore {
    dim: usize,
    chunks: Vec<Chunk>,
    data: VecData,
}

impl VectorStore {
    pub fn open(cache_dir: &Path, name: &str) -> Result<Self> {
        let (docs_path, vecs_path) = paths(cache_dir, name);
        let chunks = load_docs(&docs_path);
        anyhow::ensure!(
            !chunks.is_empty(),
            "collection `{name}` not built ({}) — run `index` first",
            docs_path.display()
        );

        #[cfg(feature = "fs")]
        let data = {
            let file = std::fs::File::open(&vecs_path)
                .with_context(|| format!("opening {}", vecs_path.display()))?;
            // SAFETY: the index is rebuilt atomically; the file is not mutated while mapped.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            VecData::Mapped(mmap)
        };
        #[cfg(not(feature = "fs"))]
        let data = VecData::Ram(super::floats_from_bytes(
            &std::fs::read(&vecs_path)
                .with_context(|| format!("reading {}", vecs_path.display()))?,
        ));

        let dim = data.as_slice().len() / chunks.len();
        Ok(Self { dim, chunks, data })
    }

    /// Iterate `(chunk, vector)` in row order.
    pub fn rows(&self) -> impl Iterator<Item = (&Chunk, &[f32])> {
        let slice = self.data.as_slice();
        let dim = self.dim;
        self.chunks
            .iter()
            .enumerate()
            .map(move |(i, c)| (c, &slice[i * dim..(i + 1) * dim]))
    }
}
