struct Hud {
    size: vec4<f32>,
    // Blend::EndPortal: atlas rects (u0, v0, width, height) of end_sky.png and
    // end_portal.png, then vanilla's GameTime in x.
    end_portal: array<vec4<f32>, 3>,
}
@group(0) @binding(0) var<uniform> hud: Hud;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;
// enchanted_glint_item on its own repeating texture (vanilla's GlintSampler).
@group(0) @binding(3) var glint_tex: texture_2d<f32>;
@group(0) @binding(4) var glint_sampler: sampler;
// The shared atlas is two textures (gfx/shared_atlas.rs): `atlas` holds its
// static rows and `atlas_dynamic` the rows below them, which the game writes
// while it runs. Atlas coordinates address both as one atlas `rows.x` rows
// tall. The dynamic rows are held in bands of `rows.y` rows, only the ones
// written to, in the order they were first written: `bands` has each band's
// place in `atlas_dynamic`, four a vector, 0xffffffff for one never written.
struct AtlasPages {
    rows: vec4<u32>,
    bands: array<vec4<u32>, 64>,
}
@group(0) @binding(5) var atlas_dynamic: texture_2d<f32>;
@group(0) @binding(6) var<uniform> atlas_pages: AtlasPages;
// The atlas's texels across and down.
fn atlas_size() -> vec2<u32> {
    return vec2(textureDimensions(atlas).x, atlas_pages.rows.x);
}
// The row of `atlas_dynamic` holding row `row` of the dynamic part, or -1 when
// its band was never written.
fn atlas_dynamic_row(row: f32) -> f32 {
    let band_rows = f32(atlas_pages.rows.y);
    let band = min(u32(row / band_rows), 255u);
    let held = atlas_pages.bands[band / 4u][band % 4u];
    if held == 0xffffffffu {
        return -1.0;
    }
    return f32(held) * band_rows + (row - f32(band) * band_rows);
}
// The atlas at `uv` through `atlas_sampler`, transparent black in a band never
// written. It has one level, so the explicit level 0 is what an implicit
// sample read, and it is valid in any control flow.
fn atlas_sample(uv: vec2<f32>) -> vec4<f32> {
    let upper = f32(textureDimensions(atlas).y);
    let row = uv.y * f32(atlas_pages.rows.x);
    if row < upper {
        return textureSampleLevel(atlas, atlas_sampler, vec2(uv.x, row / upper), 0.0);
    }
    let held = atlas_dynamic_row(row - upper);
    if held < 0.0 {
        return vec4(0.0);
    }
    return textureSampleLevel(atlas_dynamic, atlas_sampler, vec2(uv.x, held / f32(textureDimensions(atlas_dynamic).y)), 0.0);
}
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    // projection_from_position(gl_Position): the End portal's texProj0.
    @location(2) proj: vec4<f32>,
    @location(3) glint_uv: vec2<f32>,
}
@vertex fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec4<f32>, @location(3) depth: f32, @location(4) glint_uv: vec2<f32>) -> Out {
    let pixels = pos * hud.size.z;
    var out: Out;
    out.clip = vec4(pixels.x / hud.size.x * 2.0 - 1.0, 1.0 - pixels.y / hud.size.y * 2.0, depth, 1.0);
    out.uv = uv;
    out.color = color;
    out.proj = vec4(out.clip.xy * 0.5 + 0.5, out.clip.zw);
    out.glint_uv = glint_uv;
    return out;
}
@fragment fn fs_main(in: Out) -> @location(0) vec4<f32> {
    // u >= 2: the GLINT_TEXTURING sample over the item, masked by the item's
    // texture alpha (item.fsh's ALPHA_CUTOUT and GLINT path).
    if in.uv.x >= 2.0 {
        let mask = atlas_sample(in.uv - vec2(2.0, 0.0)).a;
        if mask < 0.1 { discard; }
        let glint = hud.size.w * textureSample(glint_tex, glint_sampler, in.glint_uv);
        return vec4(glint.rgb * glint.rgb, mask);
    }
    let c = atlas_sample(in.uv) * in.color;
    if c.a < 0.004 { discard; }
    return c;
}
@fragment fn fs_crosshair(in: Out) -> @location(0) vec4<f32> {
    let c = atlas_sample(in.uv) * in.color;
    if c.a == 0.0 { discard; }
    return c;
}
// rendertype_end_portal.fsh (entity.wgsl has the same for portal blocks).
const PORTAL_COLORS = array<vec3<f32>, 16>(
    vec3(0.022087, 0.098399, 0.110818),
    vec3(0.011892, 0.095924, 0.089485),
    vec3(0.027636, 0.101689, 0.100326),
    vec3(0.046564, 0.109883, 0.114838),
    vec3(0.064901, 0.117696, 0.097189),
    vec3(0.063761, 0.086895, 0.123646),
    vec3(0.084817, 0.111994, 0.166380),
    vec3(0.097489, 0.154120, 0.091064),
    vec3(0.106152, 0.131144, 0.195191),
    vec3(0.097721, 0.110188, 0.187229),
    vec3(0.133516, 0.138278, 0.148582),
    vec3(0.070006, 0.243332, 0.235792),
    vec3(0.196766, 0.142899, 0.214696),
    vec3(0.047281, 0.315338, 0.321970),
    vec3(0.204675, 0.390010, 0.302066),
    vec3(0.080955, 0.314821, 0.661491),
);
fn portal_sample(rect: vec4<f32>, p: vec4<f32>) -> vec3<f32> {
    let st = fract(p.xy / p.w);
    return atlas_sample(rect.xy + st * rect.zw).rgb;
}
fn portal_layer(p: vec4<f32>, layer: f32) -> vec4<f32> {
    let a = radians((layer * layer * 4321.0 + layer * 9.0) * 2.0);
    let c = cos(a);
    let s = sin(a);
    let k = (4.5 - layer / 4.0) * 2.0;
    var v = vec4(k * (c * p.x - s * p.y), k * (s * p.x + c * p.y), p.z, p.w);
    v = vec4(v.x + 17.0 / layer * v.w, v.y + (2.0 + layer / 1.5) * (hud.end_portal[2].x * 1.5) * v.w, v.z, v.w);
    return vec4(0.5 * v.x + 0.25 * v.w, 0.5 * v.y + 0.25 * v.w, v.z, v.w);
}
// RenderPipelines.END_PORTAL (WinScreen's poem background): end_sky and 15
// layers of end_portal.png.
@fragment fn fs_end_portal(in: Out) -> @location(0) vec4<f32> {
    var c = portal_sample(hud.end_portal[0], in.proj) * PORTAL_COLORS[0];
    for (var i = 0; i < 15; i++) {
        c += portal_sample(hud.end_portal[1], portal_layer(in.proj, f32(i + 1))) * PORTAL_COLORS[i];
    }
    return vec4(c, 1.0);
}
