//! Dense weights arranged in four-input groups so SIMD lanes compute different outputs.

use super::net::{HALF_MAX, L2, L3, L3_IN, LayerStack};

#[derive(Clone, Debug)]
pub(super) struct PackedStack {
    pub l2: [[i8; L2 * 4]; HALF_MAX / 4],
    pub l3: [[i8; L3 * 4]; L3_IN / 4],
}

impl PackedStack {
    pub fn new(stack: &LayerStack, half: usize) -> Self {
        let mut packed = Self { l2: [[0; L2 * 4]; HALF_MAX / 4], l3: [[0; L3 * 4]; L3_IN / 4] };
        pack(&stack.l2_w, &mut packed.l2[..half / 4]);
        pack(&stack.l3_w, &mut packed.l3);
        packed
    }
}

fn pack<const INPUTS: usize, const OUTPUTS: usize, const STRIDE: usize>(
    source: &[[i8; INPUTS]; OUTPUTS],
    destination: &mut [[i8; STRIDE]],
) {
    for (group, row) in destination.iter_mut().enumerate() {
        for (output, weights) in source.iter().enumerate() {
            row[output * 4..output * 4 + 4].copy_from_slice(&weights[group * 4..group * 4 + 4]);
        }
    }
}
