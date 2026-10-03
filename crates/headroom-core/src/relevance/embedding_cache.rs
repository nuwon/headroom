//! Bounded LRU cache of text embeddings.
//!
//! Tool outputs repeat across turns (the same rows, the same file chunks,
//! the same user query), and every repeat used to be re-embedded. The cache
//! keys on a 128-bit hash of the text, so it holds no text, and evicts the
//! least recently used entries in batches once it is over capacity.
//! Feature-independent so it is unit-tested without an ONNX model.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

/// Default number of cached embeddings (≈ 4096 × 384 f32 ≈ 6 MB).
pub const DEFAULT_EMBEDDING_CACHE_CAPACITY: usize = 4096;

fn key_of(text: &str) -> u128 {
    let digest = Sha256::digest(text.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    u128::from_le_bytes(bytes)
}

struct Inner {
    map: HashMap<u128, (Arc<Vec<f32>>, u64)>,
    tick: u64,
    hits: u64,
    misses: u64,
}

/// Thread-safe LRU embedding cache.
pub struct EmbeddingCache {
    capacity: usize,
    inner: Mutex<Inner>,
}

impl EmbeddingCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                tick: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// `(hits, misses, len)` counters.
    pub fn stats(&self) -> (u64, u64, usize) {
        match self.inner.lock() {
            Ok(g) => (g.hits, g.misses, g.map.len()),
            Err(_) => (0, 0, 0),
        }
    }

    /// Embeddings for `texts`, computing only the cache misses (in one
    /// `embed` call, in input order) and caching them.
    ///
    /// `embed` is called without the cache lock held, so a slow model never
    /// blocks other threads' cache hits. Two threads missing the same text
    /// both compute it; the second insert is a harmless overwrite.
    pub fn get_or_embed<E>(
        &self,
        texts: &[&str],
        embed: impl FnOnce(Vec<String>) -> Result<Vec<Vec<f32>>, E>,
    ) -> Result<Vec<Arc<Vec<f32>>>, E> {
        let keys: Vec<u128> = texts.iter().map(|t| key_of(t)).collect();
        let mut found: Vec<Option<Arc<Vec<f32>>>> = vec![None; texts.len()];
        let mut miss_idx: Vec<usize> = Vec::new();
        if let Ok(mut g) = self.inner.lock() {
            for (i, key) in keys.iter().enumerate() {
                g.tick += 1;
                let tick = g.tick;
                if let Some(entry) = g.map.get_mut(key) {
                    entry.1 = tick;
                    found[i] = Some(Arc::clone(&entry.0));
                } else {
                    miss_idx.push(i);
                }
            }
            g.hits += (texts.len() - miss_idx.len()) as u64;
            g.misses += miss_idx.len() as u64;
        } else {
            miss_idx = (0..texts.len()).collect();
        }
        if !miss_idx.is_empty() {
            // Deduplicate identical misses within the batch.
            let mut unique: Vec<usize> = Vec::new();
            let mut slot_of: HashMap<u128, usize> = HashMap::new();
            for &i in &miss_idx {
                slot_of.entry(keys[i]).or_insert_with(|| {
                    unique.push(i);
                    unique.len() - 1
                });
            }
            let computed = embed(unique.iter().map(|&i| texts[i].to_owned()).collect())?;
            let computed: Vec<Arc<Vec<f32>>> = computed.into_iter().map(Arc::new).collect();
            if let Ok(mut g) = self.inner.lock() {
                for (slot, &i) in unique.iter().enumerate() {
                    if let Some(emb) = computed.get(slot) {
                        g.tick += 1;
                        let tick = g.tick;
                        g.map.insert(keys[i], (Arc::clone(emb), tick));
                    }
                }
                Self::evict(&mut g, self.capacity);
            }
            for &i in &miss_idx {
                if let Some(emb) = slot_of.get(&keys[i]).and_then(|&s| computed.get(s)) {
                    found[i] = Some(Arc::clone(emb));
                }
            }
        }
        Ok(found
            .into_iter()
            .map(|e| e.unwrap_or_else(|| Arc::new(Vec::new())))
            .collect())
    }

    /// Drop the least recently used entries down to 90% of capacity
    /// (batched so a full cache doesn't sort on every insert).
    fn evict(g: &mut Inner, capacity: usize) {
        if g.map.len() <= capacity {
            return;
        }
        let target = capacity - capacity / 10;
        let mut stamps: Vec<(u64, u128)> = g.map.iter().map(|(k, v)| (v.1, *k)).collect();
        stamps.sort_unstable();
        let excess = g.map.len() - target;
        for (_, key) in stamps.into_iter().take(excess) {
            g.map.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn fake(texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
    }

    #[test]
    fn computes_only_misses_and_preserves_order() {
        let cache = EmbeddingCache::new(16);
        let calls = RefCell::new(Vec::<Vec<String>>::new());
        let embed = |t: Vec<String>| {
            calls.borrow_mut().push(t.clone());
            fake(t)
        };
        let first = cache.get_or_embed(&["a", "bb", "a"], embed).unwrap();
        assert_eq!(
            first.iter().map(|e| e[0]).collect::<Vec<_>>(),
            vec![1.0, 2.0, 1.0]
        );
        // Duplicate "a" embedded once.
        assert_eq!(calls.borrow()[0], vec!["a".to_owned(), "bb".to_owned()]);
        let embed2 = |t: Vec<String>| {
            calls.borrow_mut().push(t.clone());
            fake(t)
        };
        let second = cache.get_or_embed(&["bb", "ccc"], embed2).unwrap();
        assert_eq!(
            second.iter().map(|e| e[0]).collect::<Vec<_>>(),
            vec![2.0, 3.0]
        );
        assert_eq!(calls.borrow()[1], vec!["ccc".to_owned()]);
        let (hits, misses, len) = cache.stats();
        assert_eq!((hits, misses, len), (1, 4, 3));
    }

    #[test]
    fn all_hits_skip_the_model() {
        let cache = EmbeddingCache::new(4);
        cache.get_or_embed(&["x"], fake).unwrap();
        let out = cache
            .get_or_embed(&["x"], |_| -> Result<Vec<Vec<f32>>, String> {
                panic!("model must not run on a full hit")
            })
            .unwrap();
        assert_eq!(out[0][0], 1.0);
    }

    #[test]
    fn evicts_least_recently_used() {
        let cache = EmbeddingCache::new(10);
        let texts: Vec<String> = (0..10).map(|i| format!("t{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        cache.get_or_embed(&refs, fake).unwrap();
        // Touch t0 so it is most recent, then overflow.
        cache.get_or_embed(&["t0"], fake).unwrap();
        cache.get_or_embed(&["new"], fake).unwrap();
        let (_, _, len) = cache.stats();
        assert!(len <= 10);
        let calls = RefCell::new(0);
        cache
            .get_or_embed(&["t0"], |t| {
                *calls.borrow_mut() += 1;
                fake(t)
            })
            .unwrap();
        assert_eq!(*calls.borrow(), 0, "recently used entry survived eviction");
    }

    #[test]
    fn model_errors_propagate_and_cache_nothing() {
        let cache = EmbeddingCache::new(4);
        let err = cache.get_or_embed(&["x"], |_| Err::<Vec<Vec<f32>>, _>("boom"));
        assert!(err.is_err());
        assert_eq!(cache.stats().2, 0);
    }
}
