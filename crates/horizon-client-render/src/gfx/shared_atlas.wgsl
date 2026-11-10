// The shared atlas (gfx/shared_atlas.rs), appended to every shader that reads it after the
// shader's own `atlas` texture. The atlas is cut into bands of ATLAS_BAND_ROWS rows: the
// bands nothing writes while the game runs are `atlas`, in order (the static pool), and each
// band that is written is held in `atlas_dynamic` once something is written into it (the
// dynamic pool). `atlas_pages` says where each band is. Read the atlas through
// `atlas_sample`, `atlas_linear`, `atlas_load` and `atlas_size`: atlas coordinates keep
// their meaning, and a band never written reads as transparent black, as an empty slot did.
// Each samples at level 0: the atlas has one level, so that is what an implicit sample read,
// and it is valid in any control flow.
const ATLAS_BAND_ROWS: u32 = 64u;
struct AtlasPages {
    // x: the atlas's rows, 0 when `atlas` is a whole texture of its own (a GUI picture);
    // y: the rows at its top that `atlas` holds in place (its first bands, all static).
    rows: vec4<u32>,
    // Each band's place, four a vector: 0 not held, n + 1 band n of `atlas`, 0x10000 + n + 1
    // band n of `atlas_dynamic`.
    bands: array<vec4<u32>, 64>,
}
@group(0) @binding(20) var atlas_dynamic: texture_2d<f32>;
@group(0) @binding(21) var<uniform> atlas_pages: AtlasPages;

// The atlas's texels across and down.
fn atlas_size() -> vec2<u32> {
    let size = textureDimensions(atlas);
    return select(vec2(size.x, atlas_pages.rows.x), size, atlas_pages.rows.x == 0u);
}

// Where atlas row `row` is held: x the texture (0 `atlas`, 1 `atlas_dynamic`, 2 none),
// y the row there. A row past the atlas's edge is its edge's, as a sampler clamps it.
fn atlas_row(at: f32) -> vec2<f32> {
    let row = clamp(at, 0.5, f32(atlas_pages.rows.x) - 0.5);
    if row < f32(atlas_pages.rows.y) {
        return vec2(0.0, row);
    }
    let band = min(u32(row) / ATLAS_BAND_ROWS, 255u);
    let place = atlas_pages.bands[band / 4u][band % 4u];
    if place == 0u {
        return vec2(2.0, 0.0);
    }
    let inside = row - f32(band * ATLAS_BAND_ROWS);
    return vec2(f32(place >> 16u), f32((place & 0xffffu) - 1u) * f32(ATLAS_BAND_ROWS) + inside);
}

// The texel at `held` (`atlas_row`'s) at `u` through `s`.
fn atlas_held(s: sampler, u: f32, held: vec2<f32>) -> vec4<f32> {
    if held.x == 0.0 {
        return textureSampleLevel(atlas, s, vec2(u, held.y / f32(textureDimensions(atlas).y)), 0.0);
    }
    if held.x == 1.0 {
        let rows = f32(textureDimensions(atlas_dynamic).y);
        return textureSampleLevel(atlas_dynamic, s, vec2(u, held.y / rows), 0.0);
    }
    return vec4(0.0);
}

// The atlas at `uv` through `s`, for a nearest sampler: one band's texel.
fn atlas_sample(s: sampler, uv: vec2<f32>) -> vec4<f32> {
    if atlas_pages.rows.x == 0u {
        return textureSampleLevel(atlas, s, uv, 0.0);
    }
    return atlas_held(s, uv.x, atlas_row(uv.y * f32(atlas_pages.rows.x)));
}

// The atlas at `uv` through `s`, for a linear sampler: where the two rows it blends are held
// apart (a band's edge), each row is sampled at its centre and the two are blended here.
fn atlas_linear(s: sampler, uv: vec2<f32>) -> vec4<f32> {
    if atlas_pages.rows.x == 0u {
        return textureSampleLevel(atlas, s, uv, 0.0);
    }
    let row = uv.y * f32(atlas_pages.rows.x) - 0.5;
    let above = floor(row);
    let first = atlas_row(above + 0.5);
    let second = atlas_row(above + 1.5);
    if first.x == second.x && second.y == first.y + 1.0 {
        return atlas_held(s, uv.x, vec2(first.x, first.y + (row - above)));
    }
    let blend = row - above;
    return mix(atlas_held(s, uv.x, first), atlas_held(s, uv.x, second), blend);
}

// The atlas's texel at `texel` (atlas coordinates).
fn atlas_load(texel: vec2<i32>) -> vec4<f32> {
    if atlas_pages.rows.x == 0u {
        return textureLoad(atlas, texel, 0);
    }
    let held = atlas_row(f32(texel.y) + 0.5);
    let at = vec2(texel.x, i32(held.y));
    if held.x == 0.0 {
        return textureLoad(atlas, at, 0);
    }
    if held.x == 1.0 {
        return textureLoad(atlas_dynamic, at, 0);
    }
    return vec4(0.0);
}
