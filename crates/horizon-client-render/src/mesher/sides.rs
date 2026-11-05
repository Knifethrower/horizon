//! A mesh's solid quads sorted by the side they face, so that the renderer
//! can leave out the quads turned away from the eye before the GPU shades
//! their vertices.
//!
//! A quad in the plane `x = p` whose corners go counter-clockwise seen from
//! +X faces +X. From an eye at `x < p` it is seen from behind, and at
//! `x = p` edge on; the solid pipeline culls back faces, so it draws nothing
//! from there. A bucket of such quads draws nothing at all while the eye is
//! at or below the lowest of their planes, and likewise for the other five
//! axis directions. The seventh bucket holds every other quad (sloped,
//! rotated, degenerate, or whose two triangles wind opposite ways): it is
//! drawn from everywhere.
//!
//! The buckets lie one after another in the mesh, in an order chosen so that
//! the buckets an eye can see are usually one or two runs of quads: one draw
//! each. Coplanar quads facing one way keep their model order, so an overlay
//! on its base still draws in that order. Solid cutout quads are otherwise
//! ordered from the outside toward the section interior to reduce overdraw.
//!
//! A mesh's quads over opaque texels (`OPAQUE_LAYER`) come first, in buckets of their own, then
//! the others in theirs. The first can't discard a pixel, so the renderer can draw them with a
//! shader that has no `discard`: a tile-based GPU such as Mali only rejects hidden pixels early,
//! and writes depth early, for a shader like that. Drawn first, they hide what's behind the rest.
use super::{OPAQUE_LAYER, TerrainVertex};

/// The buckets in mesh order: +X, +Z, other, +Y, -Z, -X, -Y. Seen from
/// above (+Y, the commonest case), each pair of horizontal sides with a
/// common corner is then one run, or two.
const BUCKETS: [Bucket; 7] = [
    Bucket::Side(0, true),
    Bucket::Side(2, true),
    Bucket::Other,
    Bucket::Side(1, true),
    Bucket::Side(2, false),
    Bucket::Side(0, false),
    Bucket::Side(1, false),
];
const OTHER: usize = 2;
/// How far, in blocks, the eye must be behind a bucket's planes before the
/// bucket is left out. The renderer computes both the eye and the section
/// origins in `f32`, with errors far below this.
const MARGIN: f32 = 1.0 / 16.0;

#[derive(Clone, Copy)]
enum Bucket {
    /// Quads in a plane across `axis` facing its positive (`true`) or
    /// negative direction.
    Side(usize, bool),
    Other,
}

/// Where each bucket of a mesh's solid quads ends, and the planes that tell
/// whether it can face the eye.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sides {
    /// The quads over opaque texels, then the others.
    parts: [Part; 2],
}

/// One part's buckets.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Part {
    /// The solid quads up to the end of each bucket, in `BUCKETS` order, from the mesh's start.
    ends: [u32; 7],
    /// Per bucket facing a positive direction, its lowest plane; facing a
    /// negative one, its highest (unused for the other bucket).
    planes: [f32; 7],
}

impl Sides {
    /// `quads` solid quads in one bucket drawn from everywhere: for meshes
    /// that were not sorted.
    pub fn unsorted(quads: u32) -> Self {
        let mut ends = [0; 7];
        ends[OTHER..].fill(quads);
        Self {
            parts: [
                Part::default(),
                Part {
                    ends,
                    planes: [0.0; 7],
                },
            ],
        }
    }

    /// The number of solid quads the buckets hold.
    pub fn quads(&self) -> u32 {
        self.parts[1].ends[6]
    }

    /// The number of quads over opaque texels: the mesh's first.
    pub fn opaque_quads(&self) -> u32 {
        self.parts[0].ends[6]
    }

    /// Calls `draw(first, count)` for each run of quads an eye at `eye`
    /// (relative to the mesh's origin) can see the front of, nearest the
    /// start first. Empty buckets do not break a run.
    pub fn visible(&self, eye: [f32; 3], draw: impl FnMut(u32, u32)) {
        self.visible_from(eye, eye[1], draw);
    }

    /// `visible` for every eye from `eye` straight down to the height `low`: the camera and
    /// its water mirror's eye below the plane, which differ only in y.
    pub fn visible_from(&self, eye: [f32; 3], low: f32, draw: impl FnMut(u32, u32)) {
        self.visible_in(0..2, eye, low, draw);
    }

    /// `visible_from` over the quads over opaque texels only.
    pub fn opaque_from(&self, eye: [f32; 3], low: f32, draw: impl FnMut(u32, u32)) {
        self.visible_in(0..1, eye, low, draw);
    }

    /// `visible_from` over the other quads only.
    pub fn rest_from(&self, eye: [f32; 3], low: f32, draw: impl FnMut(u32, u32)) {
        self.visible_in(1..2, eye, low, draw);
    }

    fn visible_in(
        &self,
        parts: std::ops::Range<usize>,
        eye: [f32; 3],
        low: f32,
        mut draw: impl FnMut(u32, u32),
    ) {
        let mut start = if parts.start == 0 {
            0
        } else {
            self.opaque_quads()
        };
        let mut run: Option<u32> = None;
        for part in &self.parts[parts] {
            for (i, bucket) in BUCKETS.iter().enumerate() {
                let end = part.ends[i];
                if end == start {
                    continue;
                }
                let seen = match *bucket {
                    Bucket::Side(axis, true) => eye[axis] > part.planes[i] - MARGIN,
                    Bucket::Side(1, false) => low < part.planes[i] + MARGIN,
                    Bucket::Side(axis, false) => eye[axis] < part.planes[i] + MARGIN,
                    Bucket::Other => true,
                };
                match (seen, run) {
                    (true, None) => run = Some(start),
                    (false, Some(first)) => {
                        draw(first, start - first);
                        run = None;
                    }
                    _ => {}
                }
                start = end;
            }
        }
        if let Some(first) = run {
            draw(first, start - first);
        }
    }
}

/// The bucket of a quad: the axis plane it lies in and the side its
/// triangles (corners 0, 1, 2 and 0, 2, 3, as the renderer draws them) face.
fn bucket(quad: &[TerrainVertex; 4]) -> usize {
    let p = quad.map(|v| v.pos);
    for axis in 0..3 {
        if p.iter().any(|v| v[axis] != p[0][axis]) {
            continue;
        }
        // The normal's component along `axis`, twice each triangle's area.
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        let area = |a: usize, b: usize, c: usize| {
            (p[b][u] - p[a][u]) * (p[c][w] - p[a][w]) - (p[b][w] - p[a][w]) * (p[c][u] - p[a][u])
        };
        let (first, second) = (area(0, 1, 2), area(0, 2, 3));
        let positive = first >= 0.0 && second >= 0.0;
        let negative = first <= 0.0 && second <= 0.0;
        if positive == negative {
            // Opposite windings, or no area at all.
            return OTHER;
        }
        return BUCKETS
            .iter()
            .position(|b| matches!(*b, Bucket::Side(a, s) if a == axis && s == positive))
            .expect("every side has a bucket");
    }
    OTHER
}

/// Reorders `solid`'s quads: those over opaque texels first, then the others, each by bucket.
/// With any other quad in the mesh, they are also ordered front to back within each axis-facing
/// bucket; stable sorting preserves the mesher's order for coplanar overlays.
pub fn sort(solid: &mut Vec<TerrainVertex>) -> Sides {
    let (quads, rest) = solid.as_chunks::<4>();
    debug_assert!(rest.is_empty(), "solid vertices come in quads");
    let opaque = |q: &[TerrainVertex; 4]| q[0].layer & OPAQUE_LAYER != 0;
    let has_cutout = !quads.iter().all(opaque);
    let (first, second): (Vec<_>, Vec<_>) = quads.iter().copied().partition(opaque);
    let (first, parts_first) = sort_part(first, 0, has_cutout);
    let (second, parts_second) = sort_part(second, parts_first.ends[6], has_cutout);
    solid.clear();
    solid.extend(first.into_flattened());
    solid.extend(second.into_flattened());
    Sides {
        parts: [parts_first, parts_second],
    }
}

/// One part's quads sorted by bucket, and its buckets, `before` quads into the mesh.
fn sort_part(
    quads: Vec<[TerrainVertex; 4]>,
    before: u32,
    has_cutout: bool,
) -> (Vec<[TerrainVertex; 4]>, Part) {
    let buckets: Vec<u8> = quads.iter().map(|q| bucket(q) as u8).collect();
    let mut counts = [0u32; 7];
    let mut planes = [0f32; 7];
    for (quad, &b) in quads.iter().zip(&buckets) {
        let b = usize::from(b);
        if let Bucket::Side(axis, positive) = BUCKETS[b] {
            let p = quad[0].pos[axis];
            planes[b] = match (counts[b], positive) {
                (0, _) => p,
                (_, true) => planes[b].min(p),
                (_, false) => planes[b].max(p),
            };
        }
        counts[b] += 1;
    }
    let mut ends = [0u32; 7];
    let mut total = 0;
    for (end, count) in ends.iter_mut().zip(counts) {
        total += count;
        *end = total;
    }
    let mut quads = quads;
    if !quads.is_empty() && (has_cutout || counts.iter().filter(|&&c| c > 0).count() > 1) {
        let mut next: [u32; 7] = std::array::from_fn(|b| ends[b] - counts[b]);
        let mut sorted = vec![quads[0]; quads.len()];
        for (quad, &b) in quads.iter().zip(&buckets) {
            let slot = &mut next[usize::from(b)];
            sorted[*slot as usize] = *quad;
            *slot += 1;
        }
        if has_cutout {
            let mut start = 0;
            for (i, bucket) in BUCKETS.iter().enumerate() {
                let end = ends[i] as usize;
                let slice = &mut sorted[start..end];
                if let Bucket::Side(axis, positive) = *bucket {
                    slice.sort_by(|a, b| {
                        let a = a[0].pos[axis];
                        let b = b[0].pos[axis];
                        if positive {
                            b.total_cmp(&a)
                        } else {
                            a.total_cmp(&b)
                        }
                    });
                }
                start = end;
            }
        }
        quads = sorted;
    }
    (
        quads,
        Part {
            ends: ends.map(|end| before + end),
            planes,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quads over opaque texels come first, each part draws what it holds, and the two together
    /// are what the whole mesh draws.
    #[test]
    fn opaque_quads_lie_first_in_buckets_of_their_own() {
        let mut solid = cube([0.0; 3], 1);
        solid.extend(cube([2.0, 0.0, 0.0], 2 | OPAQUE_LAYER));
        solid.extend(cube([4.0, 0.0, 0.0], 3));
        let sides = sort(&mut solid);
        assert_eq!((sides.opaque_quads(), sides.quads()), (6, 18));
        let (quads, _) = solid.as_chunks::<4>();
        assert!(quads[..6].iter().all(|q| q[0].layer == 2 | OPAQUE_LAYER));
        assert!(quads[6..].iter().all(|q| q[0].layer & OPAQUE_LAYER == 0));
        let drawn = |walk: &dyn Fn(&mut dyn FnMut(u32, u32))| {
            let mut quads = Vec::new();
            walk(&mut |first, count| quads.extend(first..first + count));
            quads.sort_unstable();
            quads
        };
        for eye in [[-5.0, 9.0, 0.5], [9.0, -5.0, 9.0], [1.5, 0.5, 0.5]] {
            let all = drawn(&|d| sides.visible_from(eye, eye[1], |a, b| d(a, b)));
            let opaque = drawn(&|d| sides.opaque_from(eye, eye[1], |a, b| d(a, b)));
            let rest = drawn(&|d| sides.rest_from(eye, eye[1], |a, b| d(a, b)));
            assert!(opaque.iter().all(|&q| q < 6), "{eye:?}: {opaque:?}");
            assert!(rest.iter().all(|&q| q >= 6), "{eye:?}: {rest:?}");
            let mut both = opaque.clone();
            both.extend(&rest);
            assert_eq!(both, all, "{eye:?}");
            assert!(!opaque.is_empty() || eye == [1.5, 0.5, 0.5], "{eye:?}");
        }
    }

    /// A mesh with nothing opaque sorts as it did.
    #[test]
    fn a_mesh_without_opaque_quads_has_an_empty_first_part() {
        let mut solid = cube([0.0; 3], 1);
        let sides = sort(&mut solid);
        assert_eq!((sides.opaque_quads(), sides.quads()), (0, 6));
        let mut count = 0;
        sides.opaque_from([9.0; 3], 9.0, |_, _| count += 1);
        assert_eq!(count, 0);
        let empty = sort(&mut Vec::new());
        assert_eq!((empty.opaque_quads(), empty.quads()), (0, 0));
    }

    fn quad(corners: [[f32; 3]; 4], layer: u16) -> [TerrainVertex; 4] {
        corners.map(|p| TerrainVertex::new(p, [0.0; 2], layer, 0, 0))
    }

    /// A unit block's six faces at `origin`, wound counter-clockwise seen
    /// from outside, as block models are.
    fn cube(origin: [f32; 3], layer: u16) -> Vec<TerrainVertex> {
        let [x, y, z] = origin;
        let (a, b, c) = (x + 1.0, y + 1.0, z + 1.0);
        [
            [[a, y, z], [a, b, z], [a, b, c], [a, y, c]], // +X
            [[x, y, z], [x, y, c], [x, b, c], [x, b, z]], // -X
            [[x, b, z], [x, b, c], [a, b, c], [a, b, z]], // +Y
            [[x, y, z], [a, y, z], [a, y, c], [x, y, c]], // -Y
            [[x, y, c], [a, y, c], [a, b, c], [x, b, c]], // +Z
            [[x, y, z], [x, b, z], [a, b, z], [a, y, z]], // -Z
        ]
        .into_iter()
        .flat_map(|q| quad(q, layer))
        .collect()
    }

    fn runs(sides: &Sides, eye: [f32; 3]) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        sides.visible(eye, |first, count| out.push((first, count)));
        out
    }

    /// The quads of `solid` whose triangles face `eye`, by their layer.
    fn front_facing(solid: &[TerrainVertex], eye: [f32; 3]) -> Vec<u16> {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        solid
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|q| {
                [[0, 1, 2], [0, 2, 3]].iter().any(|t| {
                    let (e1, e2) = (sub(q[t[1]].pos, q[t[0]].pos), sub(q[t[2]].pos, q[t[0]].pos));
                    let n = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    let to_eye = sub(eye, q[t[0]].pos);
                    n[0] * to_eye[0] + n[1] * to_eye[1] + n[2] * to_eye[2] > 0.0
                })
            })
            .map(|q| q[0].layer)
            .collect()
    }

    #[test]
    fn cube_faces_go_to_their_buckets_in_order() {
        let mut solid = cube([3.0, 4.0, 5.0], 0);
        for (i, v) in solid.iter_mut().enumerate() {
            v.layer = (i / 4) as u16;
        }
        let sides = sort(&mut solid);
        assert_eq!(sides.parts[1].ends, [1, 2, 2, 3, 4, 5, 6]);
        let layers: Vec<u16> = solid.iter().step_by(4).map(|v| v.layer).collect();
        // +X, +Z, +Y, -Z, -X, -Y.
        assert_eq!(layers, [0, 4, 2, 5, 1, 3]);
        assert_eq!(sides.parts[1].planes[0], 4.0); // +X at x = 4
        assert_eq!(sides.parts[1].planes[5], 3.0); // -X at x = 3
        assert_eq!(sides.parts[1].planes[3], 5.0); // +Y at y = 5
    }

    #[test]
    fn the_drawn_runs_hold_every_quad_that_faces_the_eye() {
        // Blocks across a section, with a sloped quad and a bowtie.
        let mut solid = Vec::new();
        let mut layer = 0;
        for origin in [
            [0.0, 0.0, 0.0],
            [7.0, 3.0, 12.0],
            [15.0, 15.0, 15.0],
            [2.0, 9.0, 4.0],
        ] {
            for q in cube(origin, 0).chunks(4) {
                solid.extend(q.iter().map(|v| TerrainVertex { layer, ..*v }));
                layer += 1;
            }
        }
        let slope = [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [1.0, 2.0, 2.0],
        ];
        let bowtie = [
            [5.0, 1.0, 1.0],
            [5.0, 2.0, 2.0],
            [5.0, 1.0, 2.0],
            [5.0, 2.0, 1.0],
        ];
        for corners in [slope, bowtie] {
            solid.extend(quad(corners, layer));
            layer += 1;
        }
        let sides = sort(&mut solid);
        assert_eq!(sides.quads(), u32::from(layer));
        let eyes = [-20.0f32, -0.5, 0.0, 3.5, 8.0, 15.99, 16.0, 30.0];
        for &x in &eyes {
            for &y in &eyes {
                for &z in &eyes {
                    let eye = [x, y, z];
                    let mut drawn = Vec::new();
                    for (first, count) in runs(&sides, eye) {
                        drawn.extend(
                            solid[first as usize * 4..(first + count) as usize * 4]
                                .iter()
                                .step_by(4)
                                .map(|v| v.layer),
                        );
                    }
                    for layer in front_facing(&solid, eye) {
                        assert!(drawn.contains(&layer), "{eye:?} misses quad {layer}");
                    }
                }
            }
        }
    }

    #[test]
    fn runs_merge_across_hidden_and_empty_buckets() {
        let mut solid = cube([0.0; 3], 0);
        let sides = sort(&mut solid);
        // From above, +X and +Z: +X, +Z, (no other), +Y is one run.
        assert_eq!(runs(&sides, [8.0, 8.0, 8.0]), [(0, 3)]);
        // From below, -X and -Z: -Z, -X, -Y.
        assert_eq!(runs(&sides, [-8.0, -8.0, -8.0]), [(3, 3)]);
        // From above, -X and +Z: +Z, +Y, then -X.
        assert_eq!(runs(&sides, [-8.0, 8.0, 8.0]), [(1, 2), (4, 1)]);
        // Inside the block every face is seen from behind.
        assert!(runs(&sides, [0.5; 3]).is_empty());
        // Level with its top, the eye may see it: the margin keeps it.
        assert_eq!(runs(&sides, [0.5, 1.0, 0.5]), [(2, 1)]);
    }

    #[test]
    fn an_eye_and_its_mirror_below_see_the_faces_of_both() {
        let mut solid = cube([0.0; 3], 0);
        let sides = sort(&mut solid);
        let both = |eye: [f32; 3], low: f32| {
            let mut out = Vec::new();
            sides.visible_from(eye, low, |first, count| out.push((first, count)));
            out
        };
        // Above the block: +X, +Z and +Y; with the mirror's eye below it, -Y too.
        assert_eq!(runs(&sides, [8.0, 8.0, 8.0]), [(0, 3)]);
        assert_eq!(both([8.0, 8.0, 8.0], -8.0), [(0, 3), (5, 1)]);
        // A mirror above the block's bottom adds nothing.
        assert_eq!(both([8.0, 8.0, 8.0], 0.5), [(0, 3)]);
        // The sides depend on x and z alone.
        assert_eq!(both([-8.0, 8.0, -8.0], -8.0), [(2, 4)]);
    }

    #[test]
    fn unsorted_meshes_draw_everything() {
        let sides = Sides::unsorted(9);
        assert_eq!(sides.quads(), 9);
        assert_eq!(runs(&sides, [-100.0; 3]), [(0, 9)]);
        assert!(runs(&Sides::unsorted(0), [0.0; 3]).is_empty());
    }

    #[test]
    fn one_bucket_keeps_the_mesh_as_it_is() {
        let mut solid: Vec<_> = (0..3)
            .flat_map(|i| {
                quad(
                    [
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                    ],
                    i,
                )
            })
            .collect();
        let before = solid.clone();
        let sides = sort(&mut solid);
        assert_eq!(sides.parts[1].ends, [3; 7]);
        assert_eq!(
            solid.iter().map(|v| v.layer).collect::<Vec<_>>(),
            before.iter().map(|v| v.layer).collect::<Vec<_>>()
        );
    }

    #[test]
    fn front_to_back_sort_orders_positive_faces_and_keeps_ties_stable() {
        let mut solid: Vec<_> = [(1.0, 10), (3.0, 30), (3.0, 31)]
            .into_iter()
            .flat_map(|(x, layer)| {
                quad(
                    [[x, 0.0, 0.0], [x, 1.0, 0.0], [x, 1.0, 1.0], [x, 0.0, 1.0]],
                    layer,
                )
            })
            .collect();
        sort(&mut solid);
        let layers: Vec<u16> = solid
            .as_chunks::<4>()
            .0
            .iter()
            .map(|q| q[0].layer)
            .collect();
        assert_eq!(layers, [30, 31, 10]);
    }

    #[test]
    fn front_to_back_sort_orders_negative_faces_toward_the_eye() {
        let mut solid: Vec<_> = [(3.0, 30), (1.0, 10), (2.0, 20)]
            .into_iter()
            .flat_map(|(x, layer)| {
                quad(
                    [[x, 0.0, 0.0], [x, 0.0, 1.0], [x, 1.0, 1.0], [x, 1.0, 0.0]],
                    layer,
                )
            })
            .collect();
        sort(&mut solid);
        let layers: Vec<u16> = solid
            .as_chunks::<4>()
            .0
            .iter()
            .map(|q| q[0].layer)
            .collect();
        assert_eq!(layers, [10, 20, 30]);
    }
}
