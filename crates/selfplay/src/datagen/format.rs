//! Training-data files (written by `sim datagen`, read by `train/otto_nnue/data.py`).
//!
//! A data set `STEM` is three files:
//!
//! - `STEM.feat`: every sample's feature indices as little-endian `u16`, side to move first, then the
//!   opponent.
//! - `STEM.meta`: one 32-byte record per sample ([`META_BYTES`]):
//!
//!   | Offset | Type | Field |
//!   |---|---|---|
//!   | 0 | u64 | index of the sample's first feature in `STEM.feat` |
//!   | 8 | u8 ×4 | active features of me, of the opponent; bucket of me, of the opponent |
//!   | 12 | i8 | result for the side to move: 1 win, 0 draw, -1 loss |
//!   | 13 | u8 | teacher's controller mode (0 BUILD, 1 KILL, 2 CANCEL) |
//!   | 14 | u16 | pieces the side to move had placed |
//!   | 16 | u16 | pieces the side to move still placed before the game ended |
//!   | 18 | u16 | sibling: 0 for a decision; k ≥ 1 for its k-th sibling, bit 15 ([`CHOSEN`]) if it is the bot's move |
//!   | 20 | f32 | decision: the bot's search score of its move; sibling: the teacher's score of the child |
//!   | 24 | f32 | decision: the teacher's static evaluation (no path terms); sibling: as offset 20 |
//!   | 28 | u32 | game index within the file |
//!
//!   Siblings (`sim datagen --siblings K`) follow their decision record directly. They are one-ply children
//!   (after the lock, from the side to move's view) and share the decision's result, mode, ply and game.
//!
//! - `STEM.json`: `{"format": 2, "samples": N, "decisions": D, "features": F, "meta_bytes": 32, "settings": {..},
//!   "layout": <tetris_bot::nnue::layout_json()>}`. The layout is written verbatim: the trainer hashes it to match
//!   network files ([`tetris_bot::nnue::file::layout_hash`]).

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use tetris_bot::nnue::{Features, layout_json};

pub const META_BYTES: usize = 32;
/// 2: offset 18 holds the sibling field (it was reserved and 0 in format 1, which therefore reads as a
/// data set of decisions only).
pub const DATA_FORMAT: u32 = 2;
/// Bit of [`Sample::sib`] marking the sibling the bot played.
pub const CHOSEN: u16 = 1 << 15;

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub me: Features,
    pub opp: Features,
    pub bucket_me: u8,
    pub bucket_opp: u8,
    /// 1 win, 0 draw, -1 loss (side to move).
    pub result: i8,
    pub mode: u8,
    pub ply: u16,
    pub to_end: u16,
    pub search_score: f32,
    pub static_score: f32,
    pub game: u32,
    /// 0: a decision; k ≥ 1: its k-th sibling (| [`CHOSEN`]).
    pub sib: u16,
}

pub struct DataWriter {
    stem: PathBuf,
    meta: BufWriter<File>,
    feat: BufWriter<File>,
    offset: u64,
    samples: u64,
    decisions: u64,
    settings: serde_json::Value,
}

fn with_ext(stem: &Path, ext: &str) -> PathBuf {
    let mut s = stem.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

impl DataWriter {
    /// Create `STEM.meta` / `STEM.feat` (truncating). `settings` (how the data was generated) goes into
    /// `STEM.json`.
    pub fn create(stem: impl AsRef<Path>, settings: serde_json::Value) -> io::Result<DataWriter> {
        let stem = stem.as_ref().to_path_buf();
        if let Some(dir) = stem.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        Ok(DataWriter {
            meta: BufWriter::with_capacity(1 << 20, File::create(with_ext(&stem, "meta"))?),
            feat: BufWriter::with_capacity(1 << 20, File::create(with_ext(&stem, "feat"))?),
            stem,
            offset: 0,
            samples: 0,
            decisions: 0,
            settings,
        })
    }

    pub fn push(&mut self, s: &Sample) -> io::Result<()> {
        let mut m = [0u8; META_BYTES];
        m[0..8].copy_from_slice(&self.offset.to_le_bytes());
        m[8] = s.me.len as u8;
        m[9] = s.opp.len as u8;
        m[10] = s.bucket_me;
        m[11] = s.bucket_opp;
        m[12] = s.result as u8;
        m[13] = s.mode;
        m[14..16].copy_from_slice(&s.ply.to_le_bytes());
        m[16..18].copy_from_slice(&s.to_end.to_le_bytes());
        m[18..20].copy_from_slice(&s.sib.to_le_bytes());
        m[20..24].copy_from_slice(&s.search_score.to_le_bytes());
        m[24..28].copy_from_slice(&s.static_score.to_le_bytes());
        m[28..32].copy_from_slice(&s.game.to_le_bytes());
        self.meta.write_all(&m)?;
        for &f in s.me.as_slice().iter().chain(s.opp.as_slice()) {
            self.feat.write_all(&f.to_le_bytes())?;
        }
        self.offset += (s.me.len + s.opp.len) as u64;
        self.samples += 1;
        self.decisions += (s.sib == 0) as u64;
        Ok(())
    }

    /// Flush the data files and write `STEM.json`.
    pub fn finish(mut self) -> io::Result<u64> {
        self.meta.flush()?;
        self.feat.flush()?;
        let json = format!(
            "{{\"format\": {DATA_FORMAT}, \"samples\": {}, \"decisions\": {}, \"features\": {}, \"meta_bytes\": {META_BYTES}, \"settings\": {}, \"layout\": {}}}\n",
            self.samples,
            self.decisions,
            self.offset,
            self.settings,
            layout_json()
        );
        std::fs::write(with_ext(&self.stem, "json"), json)?;
        Ok(self.samples)
    }
}

/// The part of a stored sample the network sees, and its result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredSample {
    pub me: Vec<u16>,
    pub opp: Vec<u16>,
    pub bucket_me: u8,
    pub result: i8,
}

/// Read a data set back (tests and `sim nnue-eval`).
pub fn read_samples(stem: impl AsRef<Path>) -> io::Result<Vec<StoredSample>> {
    let stem = stem.as_ref();
    let meta = std::fs::read(with_ext(stem, "meta"))?;
    let feat = std::fs::read(with_ext(stem, "feat"))?;
    let f16 = |i: usize| u16::from_le_bytes([feat[2 * i], feat[2 * i + 1]]);
    Ok(meta
        .chunks_exact(META_BYTES)
        .map(|m| {
            let off = u64::from_le_bytes(m[0..8].try_into().unwrap()) as usize;
            let (a, b) = (m[8] as usize, m[9] as usize);
            StoredSample {
                me: (off..off + a).map(f16).collect(),
                opp: (off + a..off + a + b).map(f16).collect(),
                bucket_me: m[10],
                result: m[12] as i8,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_bot::nnue::extract;
    use tetris_sim::PlayerState;

    #[test]
    fn write_and_read_back() {
        let dir = std::env::temp_dir().join(format!("otto-nnue-data-{}", std::process::id()));
        let stem = dir.join("t");
        let mut w = DataWriter::create(&stem, serde_json::json!({ "games": 1 })).unwrap();
        let (a, b) = (PlayerState::tl(1), PlayerState::tl(2));
        let mut s = Sample {
            me: Features::default(),
            opp: Features::default(),
            bucket_me: 0,
            bucket_opp: 0,
            result: -1,
            mode: 0,
            ply: 3,
            to_end: 9,
            search_score: 1.5,
            static_score: -2.0,
            game: 0,
            sib: 0,
        };
        extract(&a, 24, &mut s.me);
        extract(&b, 25, &mut s.opp);
        w.push(&s).unwrap();
        w.push(&Sample { result: 1, ..s }).unwrap();
        assert_eq!(w.finish().unwrap(), 2);
        let back = read_samples(&stem).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].me, s.me.as_slice());
        assert_eq!(back[1].opp, s.opp.as_slice());
        assert_eq!((back[0].result, back[1].result), (-1, 1));
        let text = std::fs::read_to_string(with_ext(&stem, "json")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!((json["samples"].as_u64(), json["settings"]["games"].as_u64()), (Some(2), Some(1)));
        assert_eq!(json["layout"]["feature_set"], "rows-v1");
        std::fs::remove_dir_all(dir).ok();
    }
}
