// Shared colour constants for the Three.js scene and the puzzle overlays that
// draw on top of it. Kept in one module so a colour picked for one context
// cannot silently collide with one already used by the other -- the mapping
// quiz's guess ring used to sit almost on top of the axis colour before this
// module existed (docs/REVIEW_NOTES.md).

// Symmetry elements drawn by StaticStructureView (three_view.js): the
// highlighted axis/plane/center for the selected operation, in both analysis
// and puzzle mode.
export const SYMMETRY_ELEMENT_COLORS = {
  axis: 0x38bdf8,
  planeFill: 0xa78bfa,
  planeOutline: 0xc4b5fd,
  center: 0xef4444,
  glideArrow: 0xd6b65c,
};

// Puzzle-only marker rings (puzzle.js): the source atom, the player's guess,
// and the revealed correct target. Chosen to stay clear of every colour in
// SYMMETRY_ELEMENT_COLORS above, since the mapping quiz draws both the rings
// and the operation's symmetry element in the same view at once.
export const PUZZLE_RING_COLORS = {
  source: 0xffd23f,
  guess: 0xf472b6,
  target: 0x35c46a,
};
