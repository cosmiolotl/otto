//! The falling piece: movement, SRS+ rotation with kicks, and `all-mini+` spin detection
//! (RULES.md §4, §5).
//!
//! The engine stores `loc.y` as a float that is never an integer during play; only `floor(loc.y)` affects
//! cells, collisions, kicks and spin checks, so we store that row directly (`y`).

use crate::board::Board;
use crate::kicks::{is_tst_or_fin_kick, kicks};
use crate::piece::{Piece, Rotation, SPAWN_Y, Shape};

/// Spin classification of a lock (`"none" | "mini" | "normal"`). Ordered: `None < Mini < Normal`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Spin {
    #[default]
    None = 0,
    Mini = 1,
    Normal = 2,
}

impl Spin {
    pub const fn as_str(self) -> &'static str {
        match self {
            Spin::None => "none",
            Spin::Mini => "mini",
            Spin::Normal => "normal",
        }
    }

    #[inline]
    pub const fn is_spin(self) -> bool {
        !matches!(self, Spin::None)
    }
}

/// Piece position: `x` = `loc.x`, `y` = `floor(loc.y)` = board row of matrix row 0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Falling {
    pub piece: Piece,
    pub rot: Rotation,
    pub x: i8,
    pub y: i8,
}

/// Outcome of a successful rotation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rotated {
    pub from: Rotation,
    pub to: Rotation,
    /// Index into the kick list that succeeded, `None` for test 0 (in place).
    pub kick_index: Option<u8>,
    /// The table entry `(dx, dy-down)` that succeeded, `None` for test 0.
    pub kick: Option<(i8, i8)>,
    /// `#detectSpin` result for the new position.
    pub spin: Spin,
}

impl Falling {
    #[inline]
    pub fn spawn(piece: Piece) -> Falling {
        Falling { piece, rot: Rotation::North, x: piece.spawn_x(), y: SPAWN_Y }
    }

    #[inline]
    pub fn new(piece: Piece, rot: Rotation, x: i8, y: i8) -> Falling {
        Falling { piece, rot, x, y }
    }

    #[inline]
    pub fn shape(&self) -> &'static Shape {
        self.piece.shape(self.rot)
    }

    /// Cells sorted by `(x, y)`: the identity of a position, equal for symmetric rotations.
    pub fn sorted_cells(&self) -> [(i8, i8); 4] {
        let mut c = self.cells();
        c.sort_unstable();
        c
    }

    #[inline]
    pub fn cells(&self) -> [(i8, i8); 4] {
        self.shape().cells_at(self.x, self.y)
    }

    #[inline]
    pub fn fits(&self, board: &Board) -> bool {
        board.fits(self.shape(), self.x, self.y)
    }

    #[inline]
    fn fits_at(&self, board: &Board, dx: i8, dy: i8) -> bool {
        board.fits(self.shape(), self.x + dx, self.y + dy)
    }

    /// True if the piece cannot move down (it is resting).
    #[inline]
    pub fn is_grounded(&self, board: &Board) -> bool {
        !self.fits_at(board, 0, -1)
    }

    /// Shift by `dx` (±1) if legal (`Tetromino.moveLeft/moveRight`).
    #[inline]
    pub fn shift(&mut self, board: &Board, dx: i8) -> bool {
        if self.fits_at(board, dx, 0) {
            self.x += dx;
            true
        } else {
            false
        }
    }

    /// Shift as far as possible (`dasLeft/dasRight`). Returns true if it moved at least once.
    #[inline]
    pub fn das(&mut self, board: &Board, dx: i8) -> bool {
        let start = self.x;
        while self.fits_at(board, dx, 0) {
            self.x += dx;
        }
        self.x != start
    }

    /// Sonic drop (`Tetromino.softDrop` / the hard-drop descent). Returns true if it moved.
    #[inline]
    pub fn sonic_drop(&mut self, board: &Board) -> bool {
        let y = board.drop_y(self.shape(), self.x, self.y);
        let moved = y != self.y;
        self.y = y;
        moved
    }

    /// Fall exactly one row if the row below is free ([`crate::Key::SoftDropStep`], one gravity step).
    /// Returns true if it moved.
    #[inline]
    pub fn step_down(&mut self, board: &Board) -> bool {
        if self.fits_at(board, 0, -1) {
            self.y -= 1;
            true
        } else {
            false
        }
    }

    /// Rotate by `amount` (1 = CW, -1 = CCW, 2 = 180) with SRS+ kicks (`performKick`). On success the
    /// piece moves and the spin of the new position is returned. On failure nothing changes.
    pub fn rotate(&mut self, board: &Board, amount: i8) -> Option<Rotated> {
        let from = self.rot;
        let to = from.add(amount);
        let shape = self.piece.shape(to);
        if board.fits(shape, self.x, self.y) {
            self.rot = to;
            let spin = detect_spin(board, self, false);
            return Some(Rotated { from, to, kick_index: None, kick: None, spin });
        }
        for (i, &(kx, ky)) in kicks(self.piece, from, to).iter().enumerate() {
            // newY = ceil(loc.y) - 0.1 - ky  ⇒  floor(newY) = floor(loc.y) - ky (loc.y never integral).
            let nx = self.x + kx;
            let ny = self.y - ky;
            if board.fits(shape, nx, ny) {
                self.rot = to;
                self.x = nx;
                self.y = ny;
                let fin = is_tst_or_fin_kick(from, to, Some((kx, ky)));
                let spin = detect_spin(board, self, fin);
                return Some(Rotated { from, to, kick_index: Some(i as u8), kick: Some((kx, ky)), spin });
            }
        }
        None
    }
}

/// `isAllSpinPosition`: the piece cannot move left, right, up or down (`tetromino/index.ts:173-180`).
#[inline]
pub fn is_immobile(board: &Board, f: &Falling) -> bool {
    let sh = f.shape();
    !board.fits(sh, f.x - 1, f.y)
        && !board.fits(sh, f.x + 1, f.y)
        && !board.fits(sh, f.x, f.y + 1)
        && !board.fits(sh, f.x, f.y - 1)
}

/// T-spin corner rule (`#detectSpinFromCorners`, `src/engine/index.ts:1312-1359`; corner table
/// `kicks/data.ts:2564-2589`).
pub fn t_corner_spin(board: &Board, f: &Falling, fin_or_tst: bool) -> Spin {
    if board.fits(f.shape(), f.x, f.y - 1) {
        return Spin::None;
    }
    // Corners [cx, cy(down), frontA, frontB] around the centre (x+1, y-1).
    const CORNERS: [(i32, i32, u8, u8); 4] = [(-1, -1, 3, 0), (1, -1, 0, 1), (1, 1, 1, 2), (-1, 1, 2, 3)];
    let rot = f.rot as u8;
    let mut corners = 0;
    let mut front = 0;
    for &(cx, cy, fa, fb) in &CORNERS {
        if board.occupied(f.x as i32 + cx + 1, f.y as i32 - cy - 1) {
            corners += 1;
            if rot == fa || rot == fb {
                front += 1;
            }
        }
    }
    if corners < 3 {
        return Spin::None;
    }
    // "t" is in all-mini+ `types_mini`: mini unless both front corners are filled.
    let mut spin = if front != 2 { Spin::Mini } else { Spin::Normal };
    if fin_or_tst {
        spin = Spin::Normal;
    }
    spin
}

/// `#detectSpin` for `spinbonuses = "all-mini+"`: `max(tSpin, immobile ? mini : none)`.
#[inline]
pub fn detect_spin(board: &Board, f: &Falling, fin_or_tst: bool) -> Spin {
    let t = if f.piece == Piece::T { t_corner_spin(board, f, fin_or_tst) } else { Spin::None };
    let all = if is_immobile(board, f) { Spin::Mini } else { Spin::None };
    t.max(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rot_seq(board: &Board, f: &mut Falling, amounts: &[i8]) -> Option<Rotated> {
        let mut last = None;
        for &a in amounts {
            last = f.rotate(board, a);
        }
        last
    }

    #[test]
    fn tsd_is_normal() {
        // Classic TSD slot: a T dropped in East orientation and rotated CW into South.
        let b = Board::from_ascii(
            "
            ##........
            #...######
            ##.#######
            ",
        );
        let mut f = Falling::new(Piece::T, Rotation::East, 1, 3);
        f.sonic_drop(&b);
        let r = f.rotate(&b, 1).expect("rotates");
        assert_eq!(f.rot, Rotation::South);
        let cells = {
            let mut c = f.cells().to_vec();
            c.sort();
            c
        };
        assert_eq!(cells, vec![(1, 1), (2, 0), (2, 1), (3, 1)]);
        assert_eq!(r.spin, Spin::Normal);
    }

    #[test]
    fn t_mini_with_one_front_corner() {
        // T North resting on the floor of a 3-wide notch. Corners around the centre (2,0): front (1,1) and
        // (3,1), back (1,-1) and (3,-1) (the floor, always occupied).
        let b = Board::from_ascii(
            "
            #.........
            #...######
            ",
        );
        let f = Falling::new(Piece::T, Rotation::North, 1, 1);
        assert!(f.fits(&b));
        assert!(f.is_grounded(&b));
        assert_eq!(t_corner_spin(&b, &f, false), Spin::None); // only 2 corners
        let b2 = Board::from_ascii(
            "
            .#........
            #...######
            ",
        );
        // Filling front corner (1,1): 3 corners, 1 of them front → mini; a TST/fin kick makes it normal.
        assert_eq!(t_corner_spin(&b2, &f, false), Spin::Mini);
        assert_eq!(t_corner_spin(&b2, &f, true), Spin::Normal);
    }

    #[test]
    fn o_spin_in_covered_pocket_is_mini() {
        // Covered 2x2 pocket at columns 1-2, rows 0-1.
        let b = Board::from_ascii(
            "
            ##########
            #..#######
            #..#######
            ",
        );
        let mut f = Falling::new(Piece::O, Rotation::North, 1, 1);
        assert!(f.fits(&b));
        let r = f.rotate(&b, 1).unwrap();
        assert_eq!(r.kick_index, None);
        assert_eq!(r.spin, Spin::Mini);
    }

    #[test]
    fn failed_rotation_changes_nothing() {
        let b = Board::from_ascii(
            "
            ##########
            #.########
            #.########
            #.########
            #.########
            ",
        );
        let mut f = Falling::new(Piece::I, Rotation::East, -1, 3);
        assert!(f.fits(&b));
        let before = f;
        assert!(rot_seq(&b, &mut f, &[1]).is_none());
        assert_eq!(f, before);
    }
}
