//! The shared atlas on the GPU ([`crate::hud::HudAtlas`]) in two textures: its static
//! rows, packed once, and below them its dynamic rows (the font's glyphs, the map, skin
//! and icon slots, the on-screen keyboard's text, the animated sprites), which the game
//! writes while it runs. Coordinates address the two as one atlas as tall as both, and the
//! game's shaders read them as one (`atlas_sample` in hud.wgsl, entity.wgsl and
//! screen.wgsl), so the static part can be held in a form that takes no run-time writes.
//!
//! The static part is the ASTC bake of those rows ([`Baked`]) when there is one and the GPU
//! samples ASTC: a byte a texel instead of four. It then takes no writes at all.
//!
//! The dynamic part is held in bands of [`BAND_ROWS`] rows, only the ones written to. Most
//! of it is slots for what a session may never show (256 maps, 128 skins, 64 pack icons,
//! the on-screen keyboard's text, the glyph rows), each filled from its start. A band is
//! held from the first write into it, as the next band of a texture that grows by it, and
//! a page table says where each band is (`atlas_pages` in the shaders, [`SharedAtlas::pages`]).
//! A band never written reads as transparent black, as the empty slot did. Bind groups hold
//! the texture's view: [`SharedAtlas::dynamic`] gives it with the count of the times it was
//! replaced, so they are remade after the atlas grew.
//!
//! What samples the atlas without those shaders (a resource pack's text shader, a shader
//! pack, a native mod) gets [`SharedAtlas::combined`]: both parts in one texture, made the
//! first time it is asked for and kept up to date from then on.
use super::upload::Uploads;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use wgpu::util::DeviceExt as _;

/// The rows of a band of the dynamic part, the unit it is held in: a MiB at the
/// default resources' 4096 texels across.
pub const BAND_ROWS: u32 = 64;
/// The bands the page table maps: the dynamic part of the tallest atlas
/// (`hud::MAX_ATLAS_SIZE`). `AtlasPages.bands` in the shaders holds them four a vector.
pub const MAX_BANDS: usize = (crate::hud::MAX_ATLAS_SIZE / BAND_ROWS) as usize;
/// The page table entry of a band never written to.
const NOT_HELD: u32 = u32::MAX;
const DYNAMIC_LABEL: &str = "Shared atlas (dynamic rows)";

#[derive(Clone)]
pub struct SharedAtlas {
    pub width: u32,
    /// All rows, both parts.
    pub height: u32,
    /// The rows of the static part; the dynamic part has the rest.
    pub static_rows: u32,
    static_part: wgpu::Texture,
    pub static_view: wgpu::TextureView,
    /// The dynamic part's bands written so far, shared with every clone.
    held: Arc<Mutex<Held>>,
    /// The page table the shaders read (`AtlasPages`): the atlas's rows and
    /// [`BAND_ROWS`], then where each band of the dynamic part is held.
    pages: wgpu::Buffer,
    device: wgpu::Device,
    /// With an ASTC static part: the RGBA spill of the whole atlas, which
    /// [`Self::combined`] reads the static rows from.
    static_source: Option<PathBuf>,
    combined: Option<wgpu::Texture>,
}

/// The bands of the dynamic part that have been written to, in one texture.
struct Held {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// Each band of the dynamic part: its band in `texture`, or [`NOT_HELD`].
    bands: Vec<u32>,
    /// The bands `texture` holds.
    count: u32,
    /// How many times `texture` was replaced by a taller one.
    generation: u64,
}

/// The static rows as their ASTC bake, and the RGBA spill of the whole atlas
/// they were baked from (an ASTC texture copies into no RGBA texture).
pub struct Baked {
    pub texture: wgpu::Texture,
    pub source: PathBuf,
}

impl SharedAtlas {
    /// Textures for a `width` by `height` atlas whose first `static_rows` rows
    /// are its static part: `baked` for those when given, else an empty RGBA
    /// one. The dynamic part holds no band yet. Both parts keep at least a row.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        static_rows: u32,
        baked: Option<Baked>,
    ) -> Self {
        let height = height.max(2);
        let static_rows = static_rows.clamp(1, height - 1);
        let (static_part, static_source) = match baked {
            Some(baked) => (baked.texture, Some(baked.source)),
            None => (
                rgba_texture(device, "Shared atlas (static rows)", width, static_rows),
                None,
            ),
        };
        // `hud::HudAtlas::layout` keeps an atlas within `MAX_ATLAS_SIZE`, so
        // its dynamic part has at most `MAX_BANDS` bands.
        let bands =
            vec![NOT_HELD; ((height - static_rows).div_ceil(BAND_ROWS) as usize).min(MAX_BANDS)];
        let pages = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shared atlas pages"),
            contents: bytemuck::cast_slice(&page_table(height, &bands)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // A texture has a row at least: one stands in until a band is held.
        let texture = rgba_texture(device, DYNAMIC_LABEL, width, 1);
        Self {
            width,
            height,
            static_rows,
            static_view: static_part.create_view(&Default::default()),
            static_part,
            held: Arc::new(Mutex::new(Held {
                view: texture.create_view(&Default::default()),
                texture,
                bands,
                count: 0,
                generation: 0,
            })),
            pages,
            device: device.clone(),
            static_source,
            combined: None,
        }
    }
    fn held(&self) -> MutexGuard<'_, Held> {
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    /// Whether the static part is the ASTC bake, which takes no writes.
    pub fn compressed(&self) -> bool {
        self.static_source.is_some()
    }
    /// The first row the packed pixels still have to be uploaded from: past
    /// the static rows when their bake holds them already.
    pub fn first_upload_row(&self) -> u32 {
        if self.compressed() {
            self.static_rows
        } else {
            0
        }
    }
    /// What the GPU holds of each part, in bytes: (static, dynamic).
    pub fn bytes(&self) -> (u64, u64) {
        let rgba = |rows: u32| u64::from(self.width) * u64::from(rows) * 4;
        let upper = if self.compressed() {
            crate::astc::encoded_len(self.width, self.static_rows)
        } else {
            rgba(self.static_rows)
        };
        (upper, rgba(self.held().count * BAND_ROWS))
    }
    /// The dynamic part's bands: (held, all).
    pub fn bands(&self) -> (u32, u32) {
        let held = self.held();
        (held.count, held.bands.len() as u32)
    }
    /// The texture holding the dynamic part's bands, and how many times it was
    /// replaced: a bind group made with an older one holds the old texture.
    pub fn dynamic(&self) -> (wgpu::TextureView, u64) {
        let held = self.held();
        (held.view.clone(), held.generation)
    }
    /// How many times the dynamic part's texture was replaced by a taller one.
    pub fn generation(&self) -> u64 {
        self.held().generation
    }
    /// The page table, for the shaders' `atlas_pages` binding.
    pub fn pages(&self) -> &wgpu::Buffer {
        &self.pages
    }
    /// Hold the bands of rows `top..top + rows` of the dynamic part: the ones
    /// not held yet become the next bands of a texture taller by them, with the
    /// bands held so far copied over in `encoder`. Whether any was new.
    fn hold(
        &self,
        held: &mut Held,
        encoder: &mut wgpu::CommandEncoder,
        top: u32,
        rows: u32,
    ) -> bool {
        let before = held.count;
        let Some(range) = band_range(held.bands.len(), top, rows) else {
            return false;
        };
        if !take_bands(&mut held.bands[range], &mut held.count) {
            return false;
        }
        let texture = rgba_texture(
            &self.device,
            DYNAMIC_LABEL,
            self.width,
            held.count * BAND_ROWS,
        );
        if before > 0 {
            encoder.copy_texture_to_texture(
                copy_target(&held.texture, 0, 0),
                copy_target(&texture, 0, 0),
                wgpu::Extent3d {
                    width: self.width,
                    height: before * BAND_ROWS,
                    depth_or_array_layers: 1,
                },
            );
        }
        held.view = texture.create_view(&Default::default());
        held.texture = texture;
        held.generation += 1;
        log::info!(
            "Horizon: shared atlas dynamic rows hold {} of {} bands ({} MiB)",
            held.count,
            held.bands.len(),
            u64::from(self.width) * u64::from(held.count * BAND_ROWS) * 4 / (1024 * 1024)
        );
        true
    }
    /// Write whole rows of the atlas from row `y` (tightly packed RGBA, the full
    /// width) through the queue: the upload of its packed pixels. A band of the
    /// dynamic part with nothing in it (an empty slot) is not held for it.
    pub fn write_rows(&self, queue: &wgpu::Queue, y: u32, pixels: &[u8]) {
        let bytes_per_row = self.width * 4;
        let rows = (pixels.len() / bytes_per_row as usize) as u32;
        let rows_of = |skip: u32, rows: u32| {
            let start = (skip * bytes_per_row) as usize;
            &pixels[start..start + (rows * bytes_per_row) as usize]
        };
        let write = |texture: &wgpu::Texture, top: u32, rows: u32, data: &[u8]| {
            queue.write_texture(
                copy_target(texture, 0, top),
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: rows,
                    depth_or_array_layers: 1,
                },
            );
        };
        for (dynamic, top, skip, rows) in split_rows(self.static_rows, self.height, y, rows) {
            if !dynamic {
                if !self.compressed() {
                    write(&self.static_part, top, rows, rows_of(skip, rows));
                }
                continue;
            }
            let mut held = self.held();
            let mut changed = false;
            for (band_top, band_skip, band_rows) in split_bands(top, rows) {
                let data = rows_of(skip + band_skip, band_rows);
                let band = (band_top / BAND_ROWS) as usize;
                if held.bands.get(band) == Some(&NOT_HELD) {
                    if data.iter().all(|&byte| byte == 0) {
                        continue;
                    }
                    // Writes queued into the old texture land before this
                    // copy: the queue runs them ahead of the next submission.
                    let mut encoder = self.device.create_command_encoder(&Default::default());
                    changed |= self.hold(&mut held, &mut encoder, band_top, band_rows);
                    queue.submit([encoder.finish()]);
                }
                for (at, at_skip, at_rows) in held_spans(&held.bands, band_top, band_rows) {
                    let start = (at_skip * bytes_per_row) as usize;
                    write(
                        &held.texture,
                        at,
                        at_rows,
                        &data[start..start + (at_rows * bytes_per_row) as usize],
                    );
                }
            }
            if changed {
                queue.write_buffer(&self.pages, 16, bytemuck::cast_slice(&held.bands));
            }
        }
    }
    /// Stage a run-time write of `width` by `height` RGBA texels at `x`, `y` of
    /// the atlas into the part it falls in, and into the combined texture when
    /// there is one. A write past the atlas's edge is dropped.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self,
        uploads: &mut Uploads,
        encoder: &mut wgpu::CommandEncoder,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        if x.saturating_add(width) > self.width || y.saturating_add(height) > self.height {
            return;
        }
        let row = width as usize * 4;
        for (dynamic, top, skip, rows) in split_rows(self.static_rows, self.height, y, height) {
            let data = &rgba[skip as usize * row..(skip + rows) as usize * row];
            if !dynamic {
                if !self.compressed() {
                    uploads.texture(
                        encoder,
                        copy_target(&self.static_part, x, top),
                        width,
                        rows,
                        data,
                    );
                }
                continue;
            }
            let mut held = self.held();
            if self.hold(&mut held, encoder, top, rows) {
                uploads.buffer(encoder, &self.pages, 16, bytemuck::cast_slice(&held.bands));
            }
            for (at, at_skip, at_rows) in held_spans(&held.bands, top, rows) {
                let start = at_skip as usize * row;
                uploads.texture(
                    encoder,
                    copy_target(&held.texture, x, at),
                    width,
                    at_rows,
                    &data[start..start + at_rows as usize * row],
                );
            }
        }
        if let Some(combined) = &self.combined {
            uploads.texture(encoder, copy_target(combined, x, y), width, height, rgba);
        }
    }
    /// The whole atlas as one RGBA texture, for what samples it without the
    /// game's shaders: copied from the two parts the first time (the static
    /// rows read from the spill when they are the ASTC bake, the dynamic part's
    /// bands each to its place), then kept up to date by [`Self::write`]. It
    /// costs the memory the split and the bands save, so only those users make it.
    pub fn combined(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) -> &wgpu::Texture {
        if self.combined.is_none() {
            let texture = rgba_texture(device, "Shared atlas (whole)", self.width, self.height);
            let copy = |encoder: &mut wgpu::CommandEncoder,
                        from: wgpu::TexelCopyTextureInfo<'_>,
                        y,
                        rows| {
                encoder.copy_texture_to_texture(
                    from,
                    copy_target(&texture, 0, y),
                    wgpu::Extent3d {
                        width: self.width,
                        height: rows,
                        depth_or_array_layers: 1,
                    },
                );
            };
            match &self.static_source {
                Some(source) => {
                    if let Err(e) = self.static_rows_from(source, queue, &texture) {
                        log::warn!(
                            "Horizon: the whole shared atlas has no static rows: {}: {e}",
                            source.display()
                        );
                    }
                }
                None => copy(
                    encoder,
                    copy_target(&self.static_part, 0, 0),
                    0,
                    self.static_rows,
                ),
            }
            {
                let held = self.held();
                let dynamic_rows = self.height - self.static_rows;
                for (band, &at) in held.bands.iter().enumerate() {
                    if at == NOT_HELD {
                        continue;
                    }
                    let top = band as u32 * BAND_ROWS;
                    copy(
                        encoder,
                        copy_target(&held.texture, 0, at * BAND_ROWS),
                        self.static_rows + top,
                        BAND_ROWS.min(dynamic_rows - top),
                    );
                }
            }
            log::info!(
                "Horizon: shared atlas {}x{} made whole for a shader that samples it as one texture",
                self.width,
                self.height
            );
            self.combined = Some(texture);
        }
        self.combined.as_ref().expect("made just above")
    }
    /// The static rows of the RGBA spill at `source` into `texture`, a band at
    /// a time, so the transient copy stays a few MiB.
    fn static_rows_from(
        &self,
        source: &std::path::Path,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> std::io::Result<()> {
        use std::io::Read as _;
        let bytes_per_row = self.width * 4;
        let band = (4 * 1024 * 1024 / bytes_per_row).clamp(1, self.static_rows);
        let mut buf = vec![0u8; (band * bytes_per_row) as usize];
        let mut file = std::fs::File::open(source)?;
        let mut y = 0;
        while y < self.static_rows {
            let rows = band.min(self.static_rows - y);
            let len = (rows * bytes_per_row) as usize;
            file.read_exact(&mut buf[..len])?;
            queue.write_texture(
                copy_target(texture, 0, y),
                &buf[..len],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: rows,
                    depth_or_array_layers: 1,
                },
            );
            y += rows;
        }
        Ok(())
    }
}

/// The page table as the shaders read it (`AtlasPages`): the atlas's rows and
/// [`BAND_ROWS`], two words of padding, then each band's entry, the rest of the
/// [`MAX_BANDS`] not held.
fn page_table(height: u32, bands: &[u32]) -> Vec<u32> {
    let mut table = vec![height, BAND_ROWS, 0, 0];
    table.extend_from_slice(bands);
    table.resize(4 + MAX_BANDS, NOT_HELD);
    table
}

/// The bands rows `top..top + rows` of the dynamic part fall in, of `bands`.
fn band_range(bands: usize, top: u32, rows: u32) -> Option<std::ops::Range<usize>> {
    let first = (top / BAND_ROWS) as usize;
    let end = (top.saturating_add(rows).div_ceil(BAND_ROWS) as usize).min(bands);
    (rows > 0 && first < end).then_some(first..end)
}

/// Give each band of `bands` not held yet the next band of the texture, in
/// order, counting them in `count`. Whether any was new.
fn take_bands(bands: &mut [u32], count: &mut u32) -> bool {
    let mut taken = false;
    for band in bands.iter_mut().filter(|band| **band == NOT_HELD) {
        *band = *count;
        *count += 1;
        taken = true;
    }
    taken
}

/// Rows `top..top + rows` of the dynamic part cut at its bands: the first row
/// of each piece, how many of `rows` come before it, and its rows.
fn split_bands(top: u32, rows: u32) -> impl Iterator<Item = (u32, u32, u32)> {
    let end = top.saturating_add(rows);
    std::iter::successors(Some(top), |&y| Some((y / BAND_ROWS + 1) * BAND_ROWS))
        .take_while(move |&y| y < end)
        .map(move |y| (y, y - top, ((y / BAND_ROWS + 1) * BAND_ROWS).min(end) - y))
}

/// Where rows `top..top + rows` of the dynamic part are held: the first row of
/// each run in the texture, how many of `rows` come before it, and its rows.
/// Bands held one after the other make one run; a band not held, none.
fn held_spans(bands: &[u32], top: u32, rows: u32) -> Vec<(u32, u32, u32)> {
    let mut spans: Vec<(u32, u32, u32)> = Vec::new();
    for (y, skip, n) in split_bands(top, rows) {
        let Some(&at) = bands.get((y / BAND_ROWS) as usize) else {
            break;
        };
        if at == NOT_HELD {
            continue;
        }
        let at = at * BAND_ROWS + y % BAND_ROWS;
        match spans.last_mut() {
            Some(last) if last.0 + last.2 == at && last.1 + last.2 == skip => last.2 += n,
            _ => spans.push((at, skip, n)),
        }
    }
    spans
}

/// Rows `y..y + rows` of an atlas of `height` rows whose first `static_rows`
/// are its static part, by part: whether in the dynamic part, the first row in
/// that part, how many of `rows` come before it, and how many fall in it.
/// Rows past `height` fall nowhere.
fn split_rows(
    static_rows: u32,
    height: u32,
    y: u32,
    rows: u32,
) -> impl Iterator<Item = (bool, u32, u32, u32)> {
    let end = y.saturating_add(rows).min(height);
    let upper_end = end.min(static_rows);
    let upper = (y < upper_end).then(|| (false, y, 0, upper_end - y));
    let from = y.max(static_rows);
    let lower = (from < end).then(|| (true, from - static_rows, from - y, end - from));
    upper.into_iter().chain(lower)
}

fn rgba_texture(device: &wgpu::Device, label: &str, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        // COPY_SRC: the whole atlas is copied from the parts, the held bands
        // into a taller texture, and a shader pack's weather copies the rain
        // and snow sprites out of the whole atlas.
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn copy_target(texture: &wgpu::Texture, x: u32, y: u32) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x, y, z: 0 },
        aspect: wgpu::TextureAspect::All,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(y: u32, rows: u32) -> Vec<(bool, u32, u32, u32)> {
        // 100 rows, the first 40 static.
        split_rows(40, 100, y, rows).collect()
    }

    /// Rows go to the part they fall in, at that part's own row, and a band
    /// across the boundary is cut in two.
    #[test]
    fn rows_land_in_the_part_they_fall_in() {
        assert_eq!(split(0, 10), [(false, 0, 0, 10)]);
        assert_eq!(split(30, 10), [(false, 30, 0, 10)]);
        assert_eq!(split(40, 10), [(true, 0, 0, 10)]);
        assert_eq!(split(95, 5), [(true, 55, 0, 5)]);
        assert_eq!(split(35, 10), [(false, 35, 0, 5), (true, 0, 5, 5)]);
        assert_eq!(split(0, 100), [(false, 0, 0, 40), (true, 0, 40, 60)]);
    }

    /// Rows past the atlas's end, or none at all, write nothing.
    #[test]
    fn rows_past_the_end_fall_nowhere() {
        assert_eq!(split(95, 10), [(true, 55, 0, 5)]);
        assert!(split(100, 4).is_empty());
        assert!(split(10, 0).is_empty());
        assert!(split(50, 0).is_empty());
    }

    /// Rows are cut where a band ends.
    #[test]
    fn rows_are_cut_at_the_bands() {
        let b = BAND_ROWS;
        let cut = |top, rows| split_bands(top, rows).collect::<Vec<_>>();
        assert_eq!(cut(0, 10), [(0, 0, 10)]);
        assert_eq!(cut(b - 4, 10), [(b - 4, 0, 4), (b, 4, 6)]);
        assert_eq!(
            cut(b, 2 * b + 1),
            [(b, 0, b), (2 * b, b, b), (3 * b, 2 * b, 1)]
        );
        assert!(cut(5, 0).is_empty());
    }

    /// A band is held from the first write into it, after the ones held
    /// before, and keeps its place.
    #[test]
    fn bands_are_held_in_the_order_they_are_first_written() {
        let mut bands = vec![NOT_HELD; 8];
        let mut count = 0;
        assert!(take_bands(&mut bands[5..6], &mut count));
        assert!(take_bands(&mut bands[1..3], &mut count));
        assert!(!take_bands(&mut bands[5..6], &mut count));
        assert!(take_bands(&mut bands[2..5], &mut count));
        assert_eq!(count, 5);
        assert_eq!(bands, [NOT_HELD, 1, 2, 3, 4, 0, NOT_HELD, NOT_HELD]);
        assert_eq!(band_range(8, 0, 0), None);
        assert_eq!(band_range(8, BAND_ROWS - 1, 2), Some(0..2));
        assert_eq!(band_range(8, 7 * BAND_ROWS, 3 * BAND_ROWS), Some(7..8));
    }

    /// Rows go to where their bands are held, as one copy while the bands
    /// follow each other there, and nowhere for a band not held.
    #[test]
    fn rows_go_where_their_bands_are_held() {
        let b = BAND_ROWS;
        let bands = [2, 0, 1, NOT_HELD, 3];
        assert_eq!(held_spans(&bands, 3, 5), [(2 * b + 3, 0, 5)]);
        // Bands 1 and 2 are held 0 and 1: one run.
        assert_eq!(held_spans(&bands, b + 1, b), [(1, 0, b)]);
        // Band 0 is held after band 2: two copies.
        assert_eq!(held_spans(&bands, b - 2, 4), [(3 * b - 2, 0, 2), (0, 2, 2)]);
        // Band 3 is not held, band 4 is.
        assert_eq!(held_spans(&bands, 3 * b + 1, b), [(3 * b, b - 1, 1)]);
        assert!(held_spans(&bands, 3 * b, 4).is_empty());
        assert!(held_spans(&bands, 9 * b, 4).is_empty());
    }

    /// The table has its header and every band the shaders can address.
    #[test]
    fn the_page_table_fills_the_shaders_bands() {
        let table = page_table(5120, &[NOT_HELD, 0, 1]);
        assert_eq!(table.len(), 4 + MAX_BANDS);
        assert_eq!(table[..7], [5120, BAND_ROWS, 0, 0, NOT_HELD, 0, 1]);
        assert!(table[7..].iter().all(|&band| band == NOT_HELD));
        assert_eq!(MAX_BANDS % 4, 0);
        for shader in [
            include_str!("../hud.wgsl"),
            include_str!("../entity.wgsl"),
            include_str!("../screen.wgsl"),
        ] {
            assert!(shader.contains(&format!("bands: array<vec4<u32>, {}>", MAX_BANDS / 4)));
            assert!(shader.contains(&format!("min(u32(row / band_rows), {}u)", MAX_BANDS - 1)));
        }
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    /// The whole atlas read back from `combined`.
    fn read_whole(atlas: &mut SharedAtlas, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let (width, height) = (atlas.width, atlas.height);
        let mut encoder = device.create_command_encoder(&Default::default());
        let padded = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(padded * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let whole = atlas.combined(device, queue, &mut encoder);
        encoder.copy_texture_to_buffer(
            copy_target(whole, 0, 0),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        mapped
            .chunks(padded as usize)
            .flat_map(|row| row[..(width * 4) as usize].to_vec())
            .collect()
    }

    /// Bands are held only for what was written, the texture grows as writes
    /// reach new ones, and the atlas reads back as written wherever its bands are.
    #[test]
    fn the_dynamic_part_holds_only_the_bands_written_and_reads_back_whole() {
        let Some((device, queue)) = device() else {
            eprintln!("no adapter: skipped");
            return;
        };
        let (width, static_rows) = (64, 8);
        let height = static_rows + 5 * BAND_ROWS;
        // The packed atlas: a static row pattern, empty dynamic rows but for
        // a sprite in band 3.
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        let texel = |x: u32, y: u32| ((y * width + x) * 4) as usize;
        for y in 0..static_rows {
            pixels[texel(0, y)..texel(0, y + 1)].fill(y as u8 + 1);
        }
        let sprite = static_rows + 3 * BAND_ROWS + 10;
        pixels[texel(5, sprite)..texel(9, sprite)].fill(200);
        let mut atlas = SharedAtlas::new(&device, width, height, static_rows, None);
        atlas.write_rows(&queue, 0, &pixels);
        assert_eq!(atlas.bands(), (1, 5));
        let (_, generation) = atlas.dynamic();
        // A run-time write across bands 0 and 1 holds both, after band 3.
        let mut uploads = Uploads::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        let (x, y, w, h) = (3, static_rows + BAND_ROWS - 2, 2, 4);
        let patch: Vec<u8> = (0..w * h * 4).map(|i| i as u8 + 1).collect();
        atlas.write(&mut uploads, &mut encoder, x, y, w, h, &patch);
        uploads.finish(&encoder);
        queue.submit([encoder.finish()]);
        assert_eq!(atlas.bands(), (3, 5));
        assert_eq!(atlas.bytes().1, u64::from(width * 3 * BAND_ROWS * 4));
        assert!(atlas.dynamic().1 > generation);
        for row in 0..h {
            let at = texel(x, y + row);
            pixels[at..at + (w * 4) as usize]
                .copy_from_slice(&patch[(row * w * 4) as usize..((row + 1) * w * 4) as usize]);
        }
        assert!(read_whole(&mut atlas, &device, &queue) == pixels);
    }
}
