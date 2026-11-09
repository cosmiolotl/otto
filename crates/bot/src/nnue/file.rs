//! The `.nnue` file format (written by `train/otto_nnue/export.py`), little endian:
//!
//! ```text
//! "OTTONNUE", u32 version, u32 layout hash, u32 n_features, l1, l2, l3, n_buckets,
//! u32 desc length, desc (UTF-8),
//! ft_b i16[L1], ft_w i16[N_FEATURES · L1] (feature-major),
//! per bucket: l2_w i8[L2 · L2_IN] (output-major), l2_b i32[L2], l3_w i8[L3 · L3_IN], l3_b i32[L3],
//!             out_w i8[L3], out_b i32.
//! ```

use std::io::{self, Read, Write};
use std::path::Path;

use super::features::{self, N_BUCKETS, N_FEATURES};
use super::net::{L1, L2, L3, Network};

const MAGIC: &[u8; 8] = b"OTTONNUE";
pub const FORMAT_VERSION: u32 = 1;

/// FNV-1a of [`features::layout_json`]: identifies the feature layout in network files.
pub fn layout_hash() -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &b in features::layout_json().as_bytes() {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    h
}

/// The header fields after the magic, before the description length.
fn header() -> [u32; 7] {
    [FORMAT_VERSION, layout_hash(), N_FEATURES as u32, L1 as u32, L2 as u32, L3 as u32, N_BUCKETS as u32]
}

impl Network {
    pub fn load(path: impl AsRef<Path>) -> io::Result<Network> {
        let path = path.as_ref();
        let in_file = |e: io::Error| io::Error::new(e.kind(), format!("{}: {e}", path.display()));
        let bytes = std::fs::read(path).map_err(in_file)?;
        Network::read(&mut bytes.as_slice()).map_err(in_file)
    }

    pub fn read(r: &mut impl Read) -> io::Result<Network> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("not an otto network file".into()));
        }
        let mut got = [0u32; 7];
        for v in got.iter_mut() {
            *v = read_u32(r)?;
        }
        if got != header() {
            return Err(bad(format!(
                "architecture mismatch: file has version/layout/features/l1/l2/l3/buckets {got:?}, this build expects {:?}",
                header()
            )));
        }
        let mut desc = vec![0u8; read_u32(r)? as usize];
        r.read_exact(&mut desc)?;
        let mut n = Network::zeroed();
        n.desc = String::from_utf8_lossy(&desc).into_owned();
        read_i16s(r, &mut n.ft_b.0)?;
        read_i16s(r, &mut n.ft_w)?;
        for s in n.stacks.iter_mut() {
            for row in s.l2_w.iter_mut() {
                read_i8s(r, row)?;
            }
            read_i32s(r, &mut s.l2_b)?;
            for row in s.l3_w.iter_mut() {
                read_i8s(r, row)?;
            }
            read_i32s(r, &mut s.l3_b)?;
            read_i8s(r, &mut s.out_w)?;
            s.out_b = read_u32(r)? as i32;
        }
        let mut rest = [0u8; 1];
        if r.read(&mut rest)? != 0 {
            return Err(bad("trailing bytes after the network".into()));
        }
        Ok(n)
    }

    pub fn write(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(MAGIC)?;
        let desc = self.desc.as_bytes();
        for v in header().into_iter().chain([desc.len() as u32]) {
            w.write_all(&v.to_le_bytes())?;
        }
        w.write_all(desc)?;
        let mut buf = Vec::with_capacity(self.ft_w.len() * 2 + 4096);
        buf.extend(self.ft_b.0.iter().flat_map(|v| v.to_le_bytes()));
        buf.extend(self.ft_w.iter().flat_map(|v| v.to_le_bytes()));
        for s in &self.stacks {
            buf.extend(s.l2_w.iter().flatten().map(|&v| v as u8));
            buf.extend(s.l2_b.iter().flat_map(|v| v.to_le_bytes()));
            buf.extend(s.l3_w.iter().flatten().map(|&v| v as u8));
            buf.extend(s.l3_b.iter().flat_map(|v| v.to_le_bytes()));
            buf.extend(s.out_w.iter().map(|&v| v as u8));
            buf.extend(s.out_b.to_le_bytes());
        }
        w.write_all(&buf)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut f = io::BufWriter::new(std::fs::File::create(path)?);
        self.write(&mut f)?;
        f.flush()
    }
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i16s(r: &mut impl Read, out: &mut [i16]) -> io::Result<()> {
    let mut b = vec![0u8; out.len() * 2];
    r.read_exact(&mut b)?;
    for (o, c) in out.iter_mut().zip(b.chunks_exact(2)) {
        *o = i16::from_le_bytes([c[0], c[1]]);
    }
    Ok(())
}

fn read_i32s(r: &mut impl Read, out: &mut [i32]) -> io::Result<()> {
    let mut b = vec![0u8; out.len() * 4];
    r.read_exact(&mut b)?;
    for (o, c) in out.iter_mut().zip(b.chunks_exact(4)) {
        *o = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(())
}

fn read_i8s(r: &mut impl Read, out: &mut [i8]) -> io::Result<()> {
    let mut b = vec![0u8; out.len()];
    r.read_exact(&mut b)?;
    for (o, v) in out.iter_mut().zip(b) {
        *o = v as i8;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_round_trip() {
        let net = Network::random(11);
        let mut bytes = Vec::new();
        net.write(&mut bytes).unwrap();
        let back = Network::read(&mut bytes.as_slice()).unwrap();
        assert_eq!(back.desc, net.desc);
        assert_eq!(back.ft_b, net.ft_b);
        assert_eq!(back.ft_w, net.ft_w);
        assert_eq!(back.stacks, net.stacks);
        bytes.push(0);
        assert!(Network::read(&mut bytes.as_slice()).is_err());
        bytes[8] = 9;
        assert!(Network::read(&mut bytes.as_slice()).is_err());
    }
}
