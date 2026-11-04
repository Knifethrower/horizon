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
//! What samples the atlas without those shaders (a resource pack's text shader, a shader
//! pack, a native mod) gets [`SharedAtlas::combined`]: both parts in one texture, made the
//! first time it is asked for and kept up to date from then on.
use super::upload::Uploads;
use std::path::PathBuf;

#[derive(Clone)]
pub struct SharedAtlas {
    pub width: u32,
    /// All rows, both parts.
    pub height: u32,
    /// The rows of the static part; the dynamic part has the rest.
    pub static_rows: u32,
    static_part: wgpu::Texture,
    dynamic_part: wgpu::Texture,
    pub static_view: wgpu::TextureView,
    pub dynamic_view: wgpu::TextureView,
    /// With an ASTC static part: the RGBA spill of the whole atlas, which
    /// [`Self::combined`] reads the static rows from.
    static_source: Option<PathBuf>,
    combined: Option<wgpu::Texture>,
}

/// The static rows as their ASTC bake, and the RGBA spill of the whole atlas
/// they were baked from (an ASTC texture copies into no RGBA texture).
pub struct Baked {
    pub texture: wgpu::Texture,
    pub source: PathBuf,
}

/// The rows `y..y + rows` of the atlas that fall in one part: the part, the
/// first of them in it, and where they start among `rows`.
struct Span<'a> {
    texture: &'a wgpu::Texture,
    top: u32,
    skip: u32,
    rows: u32,
}

impl SharedAtlas {
    /// Textures for a `width` by `height` atlas whose first `static_rows` rows
    /// are its static part: `baked` for those when given, else empty RGBA ones.
    /// Both parts keep at least a row.
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
        let dynamic_part = rgba_texture(
            device,
            "Shared atlas (dynamic rows)",
            width,
            height - static_rows,
        );
        Self {
            width,
            height,
            static_rows,
            static_view: static_part.create_view(&Default::default()),
            dynamic_view: dynamic_part.create_view(&Default::default()),
            static_part,
            dynamic_part,
            static_source,
            combined: None,
        }
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
        (upper, rgba(self.height - self.static_rows))
    }
    /// The parts rows `y..y + rows` fall in that can be written: not the static
    /// part when it is the ASTC bake.
    fn spans(&self, y: u32, rows: u32) -> impl Iterator<Item = Span<'_>> {
        let compressed = self.compressed();
        split_rows(self.static_rows, self.height, y, rows)
            .filter(move |&(dynamic, ..)| dynamic || !compressed)
            .map(move |(dynamic, top, skip, rows)| Span {
                texture: if dynamic {
                    &self.dynamic_part
                } else {
                    &self.static_part
                },
                top,
                skip,
                rows,
            })
    }
    /// Write whole rows of the atlas from row `y` (tightly packed RGBA, the full
    /// width) through the queue: the upload of its packed pixels.
    pub fn write_rows(&self, queue: &wgpu::Queue, y: u32, pixels: &[u8]) {
        let bytes_per_row = self.width * 4;
        let rows = (pixels.len() / bytes_per_row as usize) as u32;
        for span in self.spans(y, rows) {
            let start = (span.skip * bytes_per_row) as usize;
            let end = start + (span.rows * bytes_per_row) as usize;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: span.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: span.top,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels[start..end],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(span.rows),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: span.rows,
                    depth_or_array_layers: 1,
                },
            );
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
        for span in self.spans(y, height) {
            let start = span.skip as usize * row;
            uploads.texture(
                encoder,
                copy_target(span.texture, x, span.top),
                width,
                span.rows,
                &rgba[start..start + span.rows as usize * row],
            );
        }
        if let Some(combined) = &self.combined {
            uploads.texture(encoder, copy_target(combined, x, y), width, height, rgba);
        }
    }
    /// The whole atlas as one RGBA texture, for what samples it without the
    /// game's shaders: copied from the two parts the first time (the static
    /// rows read from the spill when they are the ASTC bake), then kept up to
    /// date by [`Self::write`]. It costs the memory the split saves, so only
    /// those users make it.
    pub fn combined(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) -> &wgpu::Texture {
        if self.combined.is_none() {
            let texture = rgba_texture(device, "Shared atlas (whole)", self.width, self.height);
            let mut parts = vec![(
                &self.dynamic_part,
                self.static_rows,
                self.height - self.static_rows,
            )];
            match &self.static_source {
                Some(source) => {
                    if let Err(e) = self.static_rows_from(source, queue, &texture) {
                        log::warn!(
                            "Horizon: the whole shared atlas has no static rows: {}: {e}",
                            source.display()
                        );
                    }
                }
                None => parts.push((&self.static_part, 0, self.static_rows)),
            }
            for (part, y, rows) in parts {
                encoder.copy_texture_to_texture(
                    copy_target(part, 0, 0),
                    copy_target(&texture, 0, y),
                    wgpu::Extent3d {
                        width: self.width,
                        height: rows,
                        depth_or_array_layers: 1,
                    },
                );
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
        // COPY_SRC: the whole atlas is copied from the parts, and a shader
        // pack's weather copies the rain and snow sprites out of it.
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
    use super::split_rows;

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
}
