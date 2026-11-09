//! Line-clear attack, B2B charging/surge, combo and all clear (RULES.md §6, §7;
//! `src/engine/index.ts:1405-1491`, `src/engine/utils/damageCalc/index.ts`).

use crate::config::Config;
use crate::falling::Spin;

/// Up to 5 garbage events per lock: surge ×3, line attack, all clear.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Events {
    vals: [u32; 5],
    len: u8,
}

impl Events {
    pub const fn new() -> Events {
        Events { vals: [0; 5], len: 0 }
    }

    #[inline]
    pub fn push(&mut self, v: u32) {
        self.vals[self.len as usize] = v;
        self.len += 1;
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.vals[..self.len as usize]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn total(&self) -> u32 {
        self.as_slice().iter().sum()
    }
}

impl std::fmt::Debug for Events {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl std::ops::Deref for Events {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        self.as_slice()
    }
}

/// B2B and combo counters, raw engine convention: `-1` = none.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChainState {
    pub b2b: i32,
    pub combo: i32,
}

impl Default for ChainState {
    fn default() -> Self {
        ChainState { b2b: -1, combo: -1 }
    }
}

/// What a lock cleared.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ClearInfo {
    pub lines: u8,
    /// `lastSpin` at the lock.
    pub spin: Spin,
    /// Board empty after the clear.
    pub all_clear: bool,
    /// Cleared rows that contained garbage.
    pub garbage_cleared: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct AttackResult {
    /// `LockRes.rawGarbage`: events in order (surge chunks, line attack, all clear), zeros dropped.
    pub events: Events,
    /// `LockRes.surge`: surge total (0 if none released).
    pub surge: u32,
    /// The clear broke B2B (with or without surge).
    pub broke_b2b: bool,
}

/// Base attack for `lines` / spin (`garbageCalcV2` switch). Independent of the piece.
#[inline]
pub fn base_attack(lines: u8, spin: Spin) -> f64 {
    let s = spin.is_spin();
    match lines {
        0 => 0.0,
        1 => {
            if spin == Spin::Normal {
                2.0
            } else {
                0.0
            }
        }
        2 => {
            if spin == Spin::Normal {
                4.0
            } else {
                1.0
            }
        }
        3 => {
            if spin == Spin::Normal {
                6.0
            } else {
                2.0
            }
        }
        4 => {
            if s {
                10.0
            } else {
                4.0
            }
        }
        5 => {
            if s {
                12.0
            } else {
                5.0
            }
        }
        n => {
            let t = (n - 5) as f64;
            if s { 12.0 + 2.0 * t } else { 5.0 + t }
        }
    }
}

/// `garbageCalcV2(...).garbage` for the `"multiplier"` combo table, no target bonus. `b2b` and `combo`
/// are the clamped post-update counters (`max(counter, 0)`).
#[inline]
pub fn line_attack(lines: u8, spin: Spin, b2b: i32, combo: i32, chaining: bool) -> f64 {
    let mut g = base_attack(lines, spin);
    if lines > 0 && b2b > 0 {
        if chaining {
            let l = (b2b as f64 * 0.8).ln_1p();
            g += (1.0 + l).floor() + if b2b == 1 { 0.0 } else { (1.0 + l % 1.0) / 3.0 };
        } else {
            g += 1.0;
        }
    }
    if combo > 0 {
        g *= 1.0 + 0.25 * combo as f64;
        if combo > 1 {
            g = (combo as f64 * 1.25).ln_1p().max(g);
        }
    }
    g
}

/// The lock's attack. Updates `chain` (B2B/combo counters) exactly like `#lock`, then computes the
/// events: `[surge×3] ++ [floor(g·M + special)] ++ [floor(pc·M)]`, zeros removed.
pub fn attack(clear: &ClearInfo, chain: &mut ChainState, cfg: &Config, multiplier: f64) -> AttackResult {
    let lines = clear.lines;
    let spin = clear.spin.is_spin();
    let pc_b2b = match cfg.all_clear {
        Some(pc) if clear.all_clear && pc.b2b != 0 => pc.b2b,
        _ => 0,
    };

    // Counters (`index.ts:1405-1426`).
    let mut broke: Option<i32> = Some(chain.b2b);
    if lines > 0 {
        chain.combo += 1;
        if (spin || lines >= 4) && pc_b2b == 0 {
            chain.b2b += 1;
            broke = None;
        }
        if pc_b2b != 0 {
            chain.b2b += pc_b2b;
            broke = None;
        }
        if broke.is_some() {
            chain.b2b = -1;
        }
    } else {
        chain.combo = -1;
        broke = None;
    }

    // Special bonus (`:1428-1433`).
    let special = if cfg.special_bonus && clear.garbage_cleared > 0 && (spin || lines >= 4) { 1.0 } else { 0.0 };

    let g = line_attack(lines, clear.spin, chain.b2b.max(0), chain.combo.max(0), cfg.b2b_chaining);

    let mut raw = [0u32; 5];
    let mut n = 0usize;
    let mut surge = 0u32;

    // Surge goes first (`:1460-1478`).
    if let (Some(btb), Some(c)) = (broke, cfg.b2b_charging)
        && btb + 1 > c.at
    {
        surge = ((btb - c.at + c.base + 1) as f64 * multiplier).floor() as u32;
        // Math.round(surge / 3) for a non-negative integer = (surge + 1) / 3.
        let third = (surge + 1) / 3;
        raw[0] = third;
        raw[1] = third;
        raw[2] = surge - 2 * third;
        n = 3;
    }
    if g > 0.0 || special > 0.0 {
        raw[n] = (g * multiplier + special).floor() as u32;
        n += 1;
    }
    if let Some(pc) = cfg.all_clear
        && clear.all_clear
    {
        raw[n] = (pc.garbage * multiplier).floor() as u32;
        n += 1;
    }

    let mut events = Events::new();
    for &v in &raw[..n] {
        if v > 0 {
            events.push(v);
        }
    }
    AttackResult { events, surge, broke_b2b: broke.is_some() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(lines: u8, spin: Spin, pc: bool, gc: u8, b2b: i32, combo: i32) -> (Vec<u32>, ChainState, u32) {
        let mut ch = ChainState { b2b, combo };
        let r = attack(&ClearInfo { lines, spin, all_clear: pc, garbage_cleared: gc }, &mut ch, &Config::TL, 1.0);
        (r.events.to_vec(), ch, r.surge)
    }

    #[test]
    fn tsd_tst_and_tss() {
        assert_eq!(run(2, Spin::Normal, false, 0, -1, -1).0, vec![4]);
        assert_eq!(run(3, Spin::Normal, false, 0, -1, -1).0, vec![6]);
        assert_eq!(run(1, Spin::Normal, false, 0, -1, -1).0, vec![2]);
        // With B2B (counter ≥ 1 after increment): +1.
        let (ev, ch, _) = run(2, Spin::Normal, false, 0, 0, -1);
        assert_eq!(ev, vec![5]);
        assert_eq!(ch.b2b, 1);
        assert_eq!(run(3, Spin::Normal, false, 0, 2, -1).0, vec![7]);
    }

    #[test]
    fn quads() {
        assert_eq!(run(4, Spin::None, false, 0, -1, -1).0, vec![4]);
        assert_eq!(run(4, Spin::None, false, 0, 0, -1).0, vec![5]);
        // I-spin mini quad sends 10 (spin value for lines ≥ 4).
        assert_eq!(run(4, Spin::Mini, false, 0, -1, -1).0, vec![10]);
    }

    #[test]
    fn minis() {
        // Mini single 0 (nothing sent), but B2B-eligible.
        let (ev, ch, _) = run(1, Spin::Mini, false, 0, -1, -1);
        assert!(ev.is_empty());
        assert_eq!(ch.b2b, 0);
        assert_eq!(run(2, Spin::Mini, false, 0, -1, -1).0, vec![1]);
        assert_eq!(run(3, Spin::Mini, false, 0, -1, -1).0, vec![2]);
        // RULES §6.6: three mini singles in a row: [], [1], [1] (b2b 0,1,2; combo 0,1,2).
        let mut ch = ChainState::default();
        let mut out = vec![];
        for _ in 0..3 {
            let r = attack(
                &ClearInfo { lines: 1, spin: Spin::Mini, all_clear: false, garbage_cleared: 0 },
                &mut ch,
                &Config::TL,
                1.0,
            );
            out.push(r.events.to_vec());
        }
        assert_eq!(out, vec![vec![], vec![1], vec![1]]);
        assert_eq!(ch, ChainState { b2b: 2, combo: 2 });
        // Then a quad at b2b 3 / combo 3 → [8].
        let r = attack(&ClearInfo { lines: 4, ..Default::default() }, &mut ch, &Config::TL, 1.0);
        assert_eq!(r.events.to_vec(), vec![8]);
    }

    #[test]
    fn special_bonus_only_for_difficult_garbage_clears() {
        assert_eq!(run(2, Spin::Normal, false, 1, -1, -1).0, vec![5]);
        assert_eq!(run(4, Spin::None, false, 2, -1, -1).0, vec![5]);
        assert_eq!(run(2, Spin::None, false, 2, -1, -1).0, vec![1]);
        // Mini single on garbage: base 0 but the special bonus makes an event.
        assert_eq!(run(1, Spin::Mini, false, 1, -1, -1).0, vec![1]);
    }

    #[test]
    fn surge_release_chunks() {
        // RULES §7 table; the breaking clear is a plain single (adds 0).
        for (btb, chunks) in [
            (4, vec![1, 1, 2]),
            (5, vec![2, 2, 1]),
            (6, vec![2, 2, 2]),
            (7, vec![2, 2, 3]),
            (8, vec![3, 3, 2]),
            (10, vec![3, 3, 4]),
            (13, vec![4, 4, 5]),
        ] {
            let (ev, ch, surge) = run(1, Spin::None, false, 0, btb, -1);
            assert_eq!(ev, chunks, "btb {btb}");
            assert_eq!(surge, btb as u32);
            assert_eq!(ch.b2b, -1);
        }
        // Below the threshold: no surge.
        let (ev, ch, surge) = run(1, Spin::None, false, 0, 3, -1);
        assert!(ev.is_empty());
        assert_eq!((ch.b2b, surge), (-1, 0));
        // Surge goes before the breaking clear's own attack: a triple at btb 6.
        assert_eq!(run(3, Spin::None, false, 0, 6, -1).0, vec![2, 2, 2, 2]);
        // A 0-line placement never breaks B2B.
        let (ev, ch, _) = run(0, Spin::None, false, 0, 9, 3);
        assert!(ev.is_empty());
        assert_eq!(ch, ChainState { b2b: 9, combo: -1 });
        // Surge scaled by the multiplier.
        let mut ch = ChainState { b2b: 8, combo: -1 };
        let r = attack(&ClearInfo { lines: 1, ..Default::default() }, &mut ch, &Config::TL, 1.3);
        assert_eq!(r.surge, 10);
        assert_eq!(r.events.to_vec(), vec![3, 3, 4]);
    }

    #[test]
    fn all_clear() {
        // PC quad from b2b -1 on garbage rows: [5, 5]; b2b becomes 0.
        let (ev, ch, _) = run(4, Spin::None, true, 2, -1, -1);
        assert_eq!(ev, vec![5, 5]);
        assert_eq!(ch.b2b, 0);
        // PC single at b2b 3: [1, 5]; b2b 4.
        let (ev, ch, _) = run(1, Spin::None, true, 0, 3, -1);
        assert_eq!(ev, vec![1, 5]);
        assert_eq!(ch.b2b, 4);
        // PC single at b2b 6: no surge.
        let (ev, ch, surge) = run(1, Spin::None, true, 0, 6, -1);
        assert_eq!(ev, vec![1, 5]);
        assert_eq!((ch.b2b, surge), (7, 0));
        // PC with B2B: TSD PC at b2b 2 → 4+1 = 5, then 5.
        assert_eq!(run(2, Spin::Normal, true, 0, 2, -1).0, vec![5, 5]);
    }

    #[test]
    fn combo_table() {
        // RULES §6.4 table (M = 1, no special bonus).
        let rows: [(i32, u32, u32, u32); 9] = [
            (0, 0, 1, 5),
            (1, 0, 1, 6),
            (2, 1, 1, 7),
            (3, 1, 1, 8),
            (4, 1, 2, 10),
            (5, 1, 2, 11),
            (6, 2, 2, 12),
            (8, 2, 3, 15),
            (12, 2, 4, 20),
        ];
        for (combo, single, double, quad) in rows {
            let pre = combo - 1;
            let s = run(1, Spin::None, false, 0, -1, pre).0;
            assert_eq!(s.first().copied().unwrap_or(0), single, "single combo {combo}");
            let d = run(2, Spin::None, false, 0, -1, pre).0;
            assert_eq!(d.first().copied().unwrap_or(0), double, "double combo {combo}");
            let q = run(4, Spin::None, false, 0, 0, pre).0;
            assert_eq!(q, vec![quad], "quad combo {combo}");
            let t = run(2, Spin::Normal, false, 0, 0, pre).0;
            assert_eq!(t, vec![quad], "tsd combo {combo}");
        }
        assert!((line_attack(1, Spin::None, 0, 2, false) - 1.252_762_968_495_368).abs() < 1e-12);
        assert_eq!(line_attack(4, Spin::None, 1, 1, false), 6.25);
    }

    #[test]
    fn multiplier_rounding() {
        // floor(g * M + special): TSD (4) at M = 1.2 → 4.8 → 4; with special → 5.
        let mut ch = ChainState::default();
        let r = attack(
            &ClearInfo { lines: 2, spin: Spin::Normal, all_clear: false, garbage_cleared: 0 },
            &mut ch,
            &Config::TL,
            1.2,
        );
        assert_eq!(r.events.to_vec(), vec![4]);
        let mut ch = ChainState::default();
        let r = attack(
            &ClearInfo { lines: 2, spin: Spin::Normal, all_clear: false, garbage_cleared: 1 },
            &mut ch,
            &Config::TL,
            1.2,
        );
        assert_eq!(r.events.to_vec(), vec![5]);
        // PC: floor(5 * 1.2) = 6.
        let mut ch = ChainState::default();
        let r = attack(&ClearInfo { lines: 1, all_clear: true, ..Default::default() }, &mut ch, &Config::TL, 1.2);
        assert_eq!(r.events.to_vec(), vec![6]);
    }
}
