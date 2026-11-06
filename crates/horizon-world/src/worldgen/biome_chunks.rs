//! The generator's cache of chunks' biomes ([`super::generator`]).

use super::{biome::BiomeId, biome_chunk::BiomeChunk};
use std::{collections::HashMap, sync::Arc};

/// Chunks the biome cache keeps ([`BiomeChunks`]).
pub(crate) const CACHE_LIMIT: usize = 4096;
/// The most chunks' biomes [`BiomeChunks`] holds on this machine: [`CACHE_LIMIT`], or what a share of a
/// smaller machine's RAM holds. At about 6 KiB a chunk the full cache is 24 MiB, which a player walking
/// across a world reaches on a machine that doesn't have it to spare.
fn biome_cache_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let total = std::fs::read_to_string("/proc/meminfo").ok().and_then(|t| {
            t.lines().find_map(|line| {
                let rest = line.strip_prefix("MemTotal:")?;
                let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
                Some(kib * 1024)
            })
        });
        biome_limit_for(total)
    })
}
fn biome_limit_for(total: Option<u64>) -> usize {
    const RAM_SHARE: u64 = 128;
    const CHUNK_BYTES: u64 = 6 << 10;
    total.map_or(CACHE_LIMIT, |bytes| {
        usize::try_from(bytes / RAM_SHARE / CHUNK_BYTES).map_or(CACHE_LIMIT, |n| n.min(CACHE_LIMIT))
    })
}

/// The biomes of recently generated chunks, which their neighbours' noise, surface and structures read
/// again: two generations, the one filling and the one before. A chunk read from the old one moves to the
/// new; when the new one holds half of [`biome_cache_limit`], the old one goes. One map of 16,384 chunks cleared
/// all at once held about four times as much. A miss only builds the same biomes again.
#[derive(Default)]
pub(crate) struct BiomeChunks {
    now: HashMap<(i32, i32), Arc<BiomeChunk<BiomeId>>>,
    old: HashMap<(i32, i32), Arc<BiomeChunk<BiomeId>>>,
}
impl BiomeChunks {
    pub(crate) fn get(&mut self, key: &(i32, i32)) -> Option<&Arc<BiomeChunk<BiomeId>>> {
        if !self.now.contains_key(key) {
            let moved = self.old.remove(key)?;
            self.insert(*key, moved);
        }
        self.now.get(key)
    }
    pub(crate) fn insert(&mut self, key: (i32, i32), chunk: Arc<BiomeChunk<BiomeId>>) {
        if self.now.len() >= biome_cache_limit() / 2 && !self.now.contains_key(&key) {
            self.old = std::mem::take(&mut self.now);
        }
        self.now.insert(key, chunk);
    }
    pub(crate) fn len(&self) -> usize {
        self.now.len() + self.old.len()
    }
    pub(crate) fn values(&self) -> impl Iterator<Item = &Arc<BiomeChunk<BiomeId>>> {
        self.now.values().chain(self.old.values())
    }
    pub(crate) fn capacity(&self) -> usize {
        self.now.capacity() + self.old.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::position::ChunkPos;

    #[test]
    fn biome_cache_takes_a_share_of_a_small_machines_ram() {
        const GIB: u64 = 1 << 30;
        assert_eq!(biome_limit_for(None), CACHE_LIMIT);
        assert_eq!(biome_limit_for(Some(16 * GIB)), CACHE_LIMIT);
        assert_eq!(biome_limit_for(Some(GIB)), 1365);
    }

    /// The biome cache keeps a chunk that is read again across generations, and never holds more than
    /// its limit.
    #[test]
    fn biome_chunks_keep_what_is_read_again() {
        let chunk =
            |x| Arc::new(BiomeChunk::new(ChunkPos::new(x, 0), 0, 16, BiomeId(x as u32)).unwrap());
        let mut cache = BiomeChunks::default();
        let kept = chunk(-1);
        cache.insert((-1, 0), kept.clone());
        for x in 0..(CACHE_LIMIT * 3) as i32 {
            cache.insert((x, 0), chunk(x));
            // Read every so often, it moves up to the generation filling.
            if x % 512 == 0 {
                assert!(Arc::ptr_eq(cache.get(&(-1, 0)).unwrap(), &kept));
            }
            assert!(cache.len() <= CACHE_LIMIT);
        }
        assert!(Arc::ptr_eq(cache.get(&(-1, 0)).unwrap(), &kept));
        // The last chunks inserted are there; the first ones, never read again, are gone.
        let last = (CACHE_LIMIT * 3) as i32 - 1;
        assert_eq!(
            cache.get(&(last, 0)).unwrap().section(0),
            chunk(last).section(0)
        );
        assert!(cache.get(&(0, 0)).is_none());
    }
}
