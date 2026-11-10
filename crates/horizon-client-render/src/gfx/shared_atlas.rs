//! The shared atlas ([`crate::hud::HudAtlas`]) on the GPU, held in bands of [`BAND_ROWS`] rows.
//!
//! Most of the atlas never changes once it is packed. The rest is written while the game runs
//! (the font's glyph rows; the map, skin, trim, icon and mod image slots; the minimap; the
//! on-screen keyboard's text; the animated sprites), and most of that is slots a session never
//! fills. [`Bands`] says which bands are which. The bands nothing writes are the static pool,
//! one texture holding them in order. A written band is held in the dynamic pool, another
//! texture, only from the first write into it (or from the upload, when it has pixels then),
//! after the ones held before it; the pool grows by it, the bands held so far copied over on the
//! GPU. A page table ([`Parts`], `atlas_pages` in `shared_atlas.wgsl`) says where each band is,
//! and a band never written reads as transparent black, as an empty slot did.
//!
//! The static pool takes no writes, so it can be the ASTC bake of the static bands
//! (`--bake-atlas`, [`crate::astc::atlas_path`]) when the GPU samples ASTC: a byte a texel
//! instead of four.
//!
//! Atlas coordinates keep their meaning. Every shader that reads the atlas appends
//! `shared_atlas.wgsl` ([`WGSL`], [`with`]) and reads through its `atlas_sample`,
//! `atlas_linear`, `atlas_load` and `atlas_size`; its bind group layout carries the static pool
//! where it had the atlas, and the two other parts at the bindings [`layout_entries`] gives.
//! A texture that is a whole image of its own (a GUI picture, a test's atlas) binds with
//! [`Parts::whole`] and reads as before.
//!
//! What samples the atlas without those shaders (a resource pack's text shader, a shader pack,
//! a native mod) gets [`SharedAtlas::whole`]: the atlas in one texture, made the first time it
//! is asked for and kept up to date from then on. It costs what the bands save, so only they
//! make it.
use super::upload::Uploads;
use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use wgpu::util::DeviceExt as _;

/// The rows of a band, the unit the atlas is held in: a MiB at 4096 texels across.
pub const BAND_ROWS: u32 = 64;
/// The bands the page table maps: those of the tallest atlas (`hud::MAX_ATLAS_SIZE`).
/// `AtlasPages.bands` in `shared_atlas.wgsl` holds them four a vector.
pub const MAX_BANDS: usize = (crate::hud::MAX_ATLAS_SIZE / BAND_ROWS) as usize;
/// Where every layout that reads the atlas binds the dynamic pool and the page table, beside
/// its own binding of `atlas` (the static pool): `shared_atlas.wgsl` declares them there.
pub const DYNAMIC_BINDING: u32 = 20;
pub const PAGES_BINDING: u32 = 21;
/// The WGSL every shader that reads the atlas appends after its `atlas` texture.
pub const WGSL: &str = include_str!("shared_atlas.wgsl");
/// A page table entry's flag for a band of the dynamic pool.
const DYNAMIC: u32 = 1 << 16;
/// The page table's size: its header and an entry a band.
const PAGES_BYTES: u64 = 16 + 4 * MAX_BANDS as u64;
/// Rows read and written at once while uploading: the transient copy stays a few MiB.
const UPLOAD_BYTES: u32 = 4 * 1024 * 1024;

/// `source` with the atlas's WGSL appended, for a shader that reads the atlas.
pub fn with(source: &str) -> String {
    source.to_owned() + WGSL
}

/// The layout entries of the atlas's dynamic pool and page table (`shared_atlas.wgsl`'s
/// `atlas_dynamic` and `atlas_pages`), seen by `visibility`.
pub fn layout_entries(visibility: wgpu::ShaderStages) -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: DYNAMIC_BINDING,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: PAGES_BINDING,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(PAGES_BYTES),
            },
            count: None,
        },
    ]
}

/// Which bands of an atlas the game writes while it runs, and which of those hold pixels once it
/// is packed: what [`crate::hud::HudAtlas`] knows of its own layout (`HudAtlas::bands`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bands {
    width: u32,
    height: u32,
    /// Each band: whether anything writes into it while the game runs.
    written: Vec<bool>,
    /// Each band: whether it is written and holds pixels once packed.
    filled: Vec<bool>,
    /// The content key of the static bands' pixels (`astc::content_key_parts`), read while
    /// they are in memory: what names their ASTC bake.
    static_key: u64,
}

impl Bands {
    /// A `width` by `height` atlas whose rows `written` (top, bottom) are written while the game
    /// runs, packed as `pixels` (RGBA, empty when there are none).
    pub fn new(width: u32, height: u32, written: &[(u32, u32)], pixels: &[u8]) -> Self {
        let count = height.div_ceil(BAND_ROWS) as usize;
        let mut marked = vec![false; count];
        for &(top, bottom) in written {
            let bottom = bottom.min(height);
            if top < bottom {
                for band in top / BAND_ROWS..bottom.div_ceil(BAND_ROWS) {
                    marked[band as usize] = true;
                }
            }
        }
        let row = width as usize * 4;
        let rows_of = |band: usize| {
            let (top, bottom) = band_rows(height, band);
            pixels.get(top as usize * row..bottom as usize * row)
        };
        let filled = (0..count)
            .map(|band| marked[band] && rows_of(band).is_some_and(|b| b.iter().any(|&x| x != 0)))
            .collect();
        let static_key =
            crate::astc::content_key_parts((0..count).filter(|&b| !marked[b]).filter_map(rows_of));
        Self {
            width,
            height,
            written: marked,
            filled,
            static_key,
        }
    }
    /// An atlas nothing writes into in bands of its own: all of it static, run-time writes
    /// going to the static pool in place.
    pub fn all_static(width: u32, height: u32) -> Self {
        Self::new(width, height, &[], &[])
    }
    /// The static pool's rows: a band's for each static band, a short last one included.
    pub fn static_rows(&self) -> u32 {
        self.counts().0 * BAND_ROWS
    }
    /// The content key of the static bands' pixels: what names their ASTC bake
    /// ([`crate::astc::atlas_path`]).
    pub fn static_key(&self) -> u64 {
        self.static_key
    }
    /// The static pool as its texture holds it, what its ASTC bake encodes: the static bands of
    /// `pixels` (the packed atlas) in order, a short one filled to a band with transparent black.
    pub fn static_pool(&self, pixels: &[u8]) -> Vec<u8> {
        let row = self.width as usize * 4;
        let mut pool = vec![0; self.static_rows() as usize * row];
        let fixed = (0..self.written.len()).filter(|&band| !self.written[band]);
        for (place, band) in fixed.enumerate() {
            let (top, bottom) = band_rows(self.height, band);
            let (from, len) = (top as usize * row, (bottom - top) as usize * row);
            if let Some(rows) = pixels.get(from..from + len) {
                let at = place * BAND_ROWS as usize * row;
                pool[at..at + len].copy_from_slice(rows);
            }
        }
        pool
    }
    /// The bands: static, written, and written with pixels when packed.
    pub fn counts(&self) -> (u32, u32, u32) {
        let written = self.written.iter().filter(|&&w| w).count() as u32;
        let filled = self.filled.iter().filter(|&&f| f).count() as u32;
        (self.written.len() as u32 - written, written, filled)
    }
    /// The page table's entries as the atlas is uploaded: each static band at its place in the
    /// static pool, each written one that holds pixels in the dynamic pool, in order, the rest
    /// not held. Also how many bands each pool holds.
    fn entries(&self) -> (Vec<u32>, u32, u32) {
        let (mut fixed, mut held) = (0, 0);
        let entries = (0..self.written.len())
            .map(|band| {
                if !self.written[band] {
                    fixed += 1;
                    fixed
                } else if self.filled[band] {
                    held += 1;
                    DYNAMIC | held
                } else {
                    0
                }
            })
            .collect();
        (entries, fixed, held)
    }
}

/// Rows `top..bottom` of band `band` of an atlas `height` rows tall (the last one may be short).
fn band_rows(height: u32, band: usize) -> (u32, u32) {
    let top = band as u32 * BAND_ROWS;
    (top, (top + BAND_ROWS).min(height))
}

/// What [`SharedAtlas::new`] makes the atlas from: its packed pixels (empty when they were
/// spilled to a file), its size and its bands. Pixels, width and height alone make an atlas
/// that is static throughout.
pub struct Source<'a> {
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub bands: Cow<'a, Bands>,
}

impl<'a> From<(&'a [u8], u32, u32)> for Source<'a> {
    fn from((pixels, width, height): (&'a [u8], u32, u32)) -> Self {
        Self {
            pixels,
            width,
            height,
            bands: Cow::Owned(Bands::all_static(width, height)),
        }
    }
}

impl<'a, const N: usize> From<(&'a [u8; N], u32, u32)> for Source<'a> {
    fn from((pixels, width, height): (&'a [u8; N], u32, u32)) -> Self {
        (pixels.as_slice(), width, height).into()
    }
}

/// What a bind group holds besides the atlas's static pool: the dynamic pool and the page table
/// that goes with it, at [`layout_entries`]'s bindings. They are replaced together when the
/// dynamic pool grows ([`SharedAtlas::generation`]), so a group made before reads the bands as
/// they were.
#[derive(Clone)]
pub struct Parts {
    dynamic: wgpu::TextureView,
    pages: wgpu::Buffer,
    generation: u64,
}

impl Parts {
    /// The parts beside a texture that is a whole image of its own, bound where the static pool
    /// goes (a GUI picture, a test's atlas): it is read as it is.
    pub fn whole(device: &wgpu::Device) -> Self {
        Self {
            dynamic: rgba_texture(device, DYNAMIC_LABEL, 1, 1).create_view(&Default::default()),
            pages: pages_buffer(device, 0, 0, &[]),
            generation: 0,
        }
    }
    /// The bind group entries of these parts.
    pub fn entries(&self) -> [wgpu::BindGroupEntry<'_>; 2] {
        [
            wgpu::BindGroupEntry {
                binding: DYNAMIC_BINDING,
                resource: wgpu::BindingResource::TextureView(&self.dynamic),
            },
            wgpu::BindGroupEntry {
                binding: PAGES_BINDING,
                resource: self.pages.as_entire_binding(),
            },
        ]
    }
    /// How many times the dynamic pool had grown when these were made.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

const STATIC_LABEL: &str = "Shared atlas (static bands)";
const BAKED_LABEL: &str = "Shared atlas (static bands, ASTC)";
const DYNAMIC_LABEL: &str = "Shared atlas (written bands)";

/// The atlas on the GPU. Clones share it: a write through one is seen through all.
#[derive(Clone)]
pub struct SharedAtlas {
    bands: Arc<Bands>,
    static_pool: wgpu::Texture,
    static_view: wgpu::TextureView,
    /// With the static pool from its ASTC bake: the RGBA spill of the whole atlas, which
    /// [`Self::whole`] reads the static bands from (ASTC copies into no RGBA texture).
    baked_from: Option<Arc<Path>>,
    held: Arc<Mutex<Held>>,
    device: wgpu::Device,
}

/// What changes as bands are written.
struct Held {
    /// Each band's page table entry (`AtlasPages.bands`).
    entries: Vec<u32>,
    /// The bands the dynamic pool holds.
    count: u32,
    dynamic_pool: wgpu::Texture,
    parts: Parts,
    /// The whole atlas in one texture, once something asked for it.
    whole: Option<wgpu::Texture>,
    /// A write into the ASTC static pool was dropped (said once).
    dropped: bool,
}

impl SharedAtlas {
    /// The atlas `source` describes, uploaded from its pixels, or from `file` when they were
    /// spilled there (the whole atlas, RGBA): the static bands to the static pool, the written
    /// ones that hold pixels to the dynamic pool, in order. A band at a time is read, so the
    /// transient copy stays a few MiB. With the spill there and a GPU that samples ASTC, the
    /// static pool is the ASTC bake beside it ([`crate::astc::atlas_path`]) when there is one,
    /// and only the written bands are read.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &Source<'_>,
        file: Option<&Path>,
    ) -> Result<Self, String> {
        let (width, height) = (source.width, source.height);
        let bands = source.bands.clone().into_owned();
        if (bands.width, bands.height) != (width, height) {
            return Err(format!(
                "the shared atlas is {width}x{height}, its bands are for {}x{}",
                bands.width, bands.height
            ));
        }
        let (entries, fixed, count) = bands.entries();
        let static_rows = fixed * BAND_ROWS;
        let baked = file.filter(|_| static_rows > 0).and_then(|spill| {
            let path =
                crate::astc::atlas_path(spill.parent()?, width, static_rows, bands.static_key);
            let texture = crate::astc::load(device, queue, &path, width, static_rows, BAKED_LABEL);
            if texture.is_none() && crate::astc::supported(device) {
                log::info!(
                    "Horizon: no ASTC bake {} for the shared atlas's static bands, uploading RGBA",
                    path.display()
                );
            }
            Some((texture?, Arc::<Path>::from(spill)))
        });
        let (static_pool, baked_from) = match baked {
            Some((texture, spill)) => (texture, Some(spill)),
            None => (
                rgba_texture(device, STATIC_LABEL, width, static_rows.max(1)),
                None,
            ),
        };
        let dynamic_pool = rgba_texture(device, DYNAMIC_LABEL, width, (count * BAND_ROWS).max(1));
        let started = std::time::Instant::now();
        let mut file = match (source.pixels.is_empty(), file) {
            (false, _) => None,
            (true, Some(path)) => Some((
                std::fs::File::open(path)
                    .map_err(|e| format!("the shared atlas at {}: {e}", path.display()))?,
                path,
            )),
            (true, None) => {
                return Err(
                    "the shared atlas has no CPU pixels left and no previous texture to reuse"
                        .into(),
                );
            }
        };
        let bytes_per_row = width * 4;
        let mut buf = Vec::new();
        // The bake holds the static bands already.
        let held_runs = runs(height, &entries);
        for &(pool, place, top, bottom) in held_runs.iter().filter(|r| r.0 || baked_from.is_none())
        {
            let texture = if pool { &dynamic_pool } else { &static_pool };
            let chunk = (UPLOAD_BYTES / bytes_per_row).max(1);
            let mut y = top;
            while y < bottom {
                let rows = chunk.min(bottom - y);
                let (start, len) = (
                    y as usize * bytes_per_row as usize,
                    (rows * bytes_per_row) as usize,
                );
                let pixels = match &mut file {
                    None => source
                        .pixels
                        .get(start..start + len)
                        .ok_or("the shared atlas has fewer pixels than its size")?,
                    Some((file, path)) => {
                        use std::io::{Read as _, Seek as _, SeekFrom};
                        buf.resize(len, 0);
                        file.seek(SeekFrom::Start(start as u64))
                            .and_then(|_| file.read_exact(&mut buf))
                            .map_err(|e| format!("the shared atlas at {}: {e}", path.display()))?;
                        &buf[..]
                    }
                };
                queue.write_texture(
                    copy_target(texture, 0, place + y - top),
                    pixels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(rows),
                    },
                    wgpu::Extent3d {
                        width,
                        height: rows,
                        depth_or_array_layers: 1,
                    },
                );
                queue.submit([]);
                y += rows;
            }
        }
        let (static_bands, written, _) = bands.counts();
        let mib = |rows: u32| u64::from(width) * u64::from(rows) * 4 / (1024 * 1024);
        log::info!(
            "Horizon: shared atlas {width}x{height} {} in {} ms: {static_bands} static {BAND_ROWS}-row bands {} ({} MiB), {count} of {written} written ones held ({} MiB)",
            match &file {
                Some((_, path)) => format!("streamed from {}", path.display()),
                None => "uploaded".to_owned(),
            },
            started.elapsed().as_millis(),
            if baked_from.is_some() {
                "from their ASTC bake"
            } else {
                "RGBA"
            },
            if baked_from.is_some() {
                crate::astc::encoded_len(width, static_rows) / (1024 * 1024)
            } else {
                mib(static_rows)
            },
            mib(count * BAND_ROWS),
        );
        let parts = Parts {
            dynamic: dynamic_pool.create_view(&Default::default()),
            pages: pages_buffer(device, height, prefix_rows(height, &entries), &entries),
            generation: 0,
        };
        Ok(Self {
            bands: Arc::new(bands),
            static_view: static_pool.create_view(&Default::default()),
            static_pool,
            baked_from,
            held: Arc::new(Mutex::new(Held {
                entries,
                count,
                dynamic_pool,
                parts,
                whole: None,
                dropped: false,
            })),
            device: device.clone(),
        })
    }
    fn held(&self) -> MutexGuard<'_, Held> {
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    /// The atlas's size, both pools together.
    pub fn size(&self) -> (u32, u32) {
        (self.bands.width, self.bands.height)
    }
    /// Whether this atlas is the one `source` describes, so a new renderer can take it over.
    pub fn matches(&self, source: &Source<'_>) -> bool {
        *self.bands == *source.bands
    }
    /// The static pool, for each shader's `atlas` binding.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.static_view
    }
    /// The dynamic pool and the page table as they are now.
    pub fn parts(&self) -> Parts {
        self.held().parts.clone()
    }
    /// How many times the dynamic pool grew: a bind group whose [`Parts`] are older reads the
    /// bands as they were, and is made again.
    pub fn generation(&self) -> u64 {
        self.held().parts.generation
    }
    /// Whether the static pool is the ASTC bake of the static bands.
    pub fn compressed(&self) -> bool {
        self.baked_from.is_some()
    }
    /// What the GPU holds of each pool, in bytes: (static, dynamic).
    pub fn bytes(&self) -> (u64, u64) {
        let (width, rows) = (self.bands.width, self.bands.static_rows());
        let rgba = |rows: u32| u64::from(width) * u64::from(rows) * 4;
        let fixed = if self.compressed() {
            crate::astc::encoded_len(width, rows)
        } else {
            rgba(rows)
        };
        (fixed, rgba(self.held().count * BAND_ROWS))
    }
    /// The bands of the dynamic pool: (held, written).
    pub fn held_bands(&self) -> (u32, u32) {
        (self.held().count, self.bands.counts().1)
    }
    /// Stage a run-time write of `width` by `height` RGBA texels at `x`, `y` of the atlas, into
    /// the pool each of its bands is in, and into the whole atlas when there is one. A written
    /// band it reaches is held from now on, unless all it writes there is transparent black.
    /// A write past the atlas's edge is dropped, and so is one into a static band of the ASTC
    /// static pool (nothing the game writes is in one).
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
        let (atlas_width, atlas_height) = self.size();
        if width == 0
            || height == 0
            || x.saturating_add(width) > atlas_width
            || y.saturating_add(height) > atlas_height
            || rgba.len() < (width * height * 4) as usize
        {
            return;
        }
        let row = width as usize * 4;
        let mut held = self.held();
        let new: Vec<usize> = pieces(y, height)
            .filter(|&(band, skip, rows)| {
                held.entries[band] == 0
                    && rgba[skip as usize * row..(skip + rows) as usize * row]
                        .iter()
                        .any(|&byte| byte != 0)
            })
            .map(|(band, ..)| band)
            .collect();
        if !new.is_empty() {
            self.grow(&mut held, encoder, &new);
        }
        for (pool, at, skip, rows) in spans(&held.entries, y, height) {
            if !pool && self.compressed() {
                if !held.dropped {
                    log::warn!(
                        "Horizon: a run-time write at {x},{y} reached the shared atlas's ASTC static bands: dropped"
                    );
                    held.dropped = true;
                }
                continue;
            }
            let texture = if pool {
                &held.dynamic_pool
            } else {
                &self.static_pool
            };
            let data = &rgba[skip as usize * row..(skip + rows) as usize * row];
            uploads.texture(encoder, copy_target(texture, x, at), width, rows, data);
            // The whole atlas follows what is written.
            if let Some(whole) = &held.whole {
                uploads.texture(encoder, copy_target(whole, x, y + skip), width, rows, data);
            }
        }
    }
    /// Hold the written bands `new` in the dynamic pool, after the ones held before: a pool
    /// taller by them, the bands held so far copied over in `encoder`, and a page table that
    /// says so.
    fn grow(&self, held: &mut Held, encoder: &mut wgpu::CommandEncoder, new: &[usize]) {
        let (width, height) = self.size();
        let before = held.count;
        for &band in new {
            held.count += 1;
            held.entries[band] = DYNAMIC | held.count;
        }
        let pool = rgba_texture(&self.device, DYNAMIC_LABEL, width, held.count * BAND_ROWS);
        if before > 0 {
            encoder.copy_texture_to_texture(
                copy_target(&held.dynamic_pool, 0, 0),
                copy_target(&pool, 0, 0),
                wgpu::Extent3d {
                    width,
                    height: before * BAND_ROWS,
                    depth_or_array_layers: 1,
                },
            );
        }
        held.parts = Parts {
            dynamic: pool.create_view(&Default::default()),
            pages: pages_buffer(
                &self.device,
                height,
                prefix_rows(height, &held.entries),
                &held.entries,
            ),
            generation: held.parts.generation + 1,
        };
        held.dynamic_pool = pool;
        log::info!(
            "Horizon: shared atlas holds {} of {} written bands ({} MiB)",
            held.count,
            self.bands.counts().1,
            u64::from(width) * u64::from(held.count * BAND_ROWS) * 4 / (1024 * 1024)
        );
    }
    /// The whole atlas as one RGBA texture, for what samples it without the atlas's WGSL: copied
    /// from the pools in `encoder` the first time (the static bands read from the spill through
    /// `queue` when they are the ASTC bake), then kept up to date by [`Self::write`].
    pub fn whole(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        let mut held = self.held();
        if let Some(whole) = &held.whole {
            return whole.clone();
        }
        let (width, height) = self.size();
        let texture = rgba_texture(device, "Shared atlas (whole)", width, height);
        for (pool, place, top, bottom) in runs(height, &held.entries) {
            if let Some(spill) = self.baked_from.as_ref().filter(|_| !pool) {
                if let Err(e) = rows_from(spill, queue, &texture, top, bottom) {
                    log::warn!(
                        "Horizon: the whole shared atlas has no static bands from {}: {e}",
                        spill.display()
                    );
                }
                continue;
            }
            let from = if pool {
                &held.dynamic_pool
            } else {
                &self.static_pool
            };
            encoder.copy_texture_to_texture(
                copy_target(from, 0, place),
                copy_target(&texture, 0, top),
                wgpu::Extent3d {
                    width,
                    height: bottom - top,
                    depth_or_array_layers: 1,
                },
            );
        }
        log::info!(
            "Horizon: shared atlas {width}x{height} made whole for a shader that samples it as one texture"
        );
        held.whole = Some(texture.clone());
        texture
    }
}

/// Rows `top..bottom` of the RGBA spill of an atlas at `path` into the same rows of `texture`,
/// a few MiB at a time.
fn rows_from(
    path: &Path,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    top: u32,
    bottom: u32,
) -> std::io::Result<()> {
    use std::io::{Read as _, Seek as _, SeekFrom};
    let bytes_per_row = texture.width() * 4;
    let chunk = (UPLOAD_BYTES / bytes_per_row).max(1);
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(u64::from(top) * u64::from(bytes_per_row)))?;
    let mut buf = Vec::new();
    let mut y = top;
    while y < bottom {
        let rows = chunk.min(bottom - y);
        buf.resize((rows * bytes_per_row) as usize, 0);
        file.read_exact(&mut buf)?;
        queue.write_texture(
            copy_target(texture, 0, y),
            &buf,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: texture.width(),
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        y += rows;
    }
    Ok(())
}

/// The rows a band starts at in the pool its page table entry `place` names: (in the dynamic
/// pool, row), `None` when it is not held.
fn place_of(place: u32) -> Option<(bool, u32)> {
    let n = place & (DYNAMIC - 1);
    (n > 0).then(|| (place & DYNAMIC != 0, (n - 1) * BAND_ROWS))
}

/// The held bands of an atlas `height` rows tall with page table `entries`, in runs that follow
/// each other in the atlas and in one pool: (in the dynamic pool, its first row there, the
/// atlas's rows top..bottom).
fn runs(height: u32, entries: &[u32]) -> Vec<(bool, u32, u32, u32)> {
    let mut runs: Vec<(bool, u32, u32, u32)> = Vec::new();
    for (band, &entry) in entries.iter().enumerate() {
        let Some((pool, place)) = place_of(entry) else {
            continue;
        };
        let (top, bottom) = band_rows(height, band);
        match runs.last_mut() {
            Some(run) if run.0 == pool && run.3 == top && run.1 + (top - run.2) == place => {
                run.3 = bottom;
            }
            _ => runs.push((pool, place, top, bottom)),
        }
    }
    runs
}

/// Rows `y..y + rows` cut at the bands: (band, how many of the rows come before, rows in it).
fn pieces(y: u32, rows: u32) -> impl Iterator<Item = (usize, u32, u32)> {
    let end = y.saturating_add(rows);
    std::iter::successors(Some(y), |&at| Some((at / BAND_ROWS + 1) * BAND_ROWS))
        .take_while(move |&at| at < end)
        .map(move |at| {
            let next = ((at / BAND_ROWS + 1) * BAND_ROWS).min(end);
            ((at / BAND_ROWS) as usize, at - y, next - at)
        })
}

/// Where rows `y..y + rows` of the atlas are held, as copies: (in the dynamic pool, the first
/// row there, how many of the rows come before, rows). Bands held one after the other in one
/// pool make one copy; a band not held, none.
fn spans(entries: &[u32], y: u32, rows: u32) -> Vec<(bool, u32, u32, u32)> {
    let mut spans: Vec<(bool, u32, u32, u32)> = Vec::new();
    for (band, skip, n) in pieces(y, rows) {
        let Some((pool, place)) = entries.get(band).and_then(|&e| place_of(e)) else {
            continue;
        };
        let at = place + (y + skip) % BAND_ROWS;
        match spans.last_mut() {
            Some(last) if last.0 == pool && last.1 + last.3 == at && last.2 + last.3 == skip => {
                last.3 += n;
            }
            _ => spans.push((pool, at, skip, n)),
        }
    }
    spans
}

/// The rows at the top of an atlas `height` rows tall that the static pool holds in place: its
/// first bands, while each is static (`atlas_pages.rows.y`).
fn prefix_rows(height: u32, entries: &[u32]) -> u32 {
    let bands = entries
        .iter()
        .enumerate()
        .take_while(|&(band, &entry)| entry == band as u32 + 1)
        .count() as u32;
    (bands * BAND_ROWS).min(height)
}

/// The page table as `shared_atlas.wgsl` reads it (`AtlasPages`): the atlas's rows, the rows the
/// static pool holds in place, two words of padding, then each band's entry, the rest of the
/// [`MAX_BANDS`] not held.
fn page_table(height: u32, prefix: u32, entries: &[u32]) -> Vec<u32> {
    let mut table = vec![height, prefix, 0, 0];
    table.extend_from_slice(&entries[..entries.len().min(MAX_BANDS)]);
    table.resize(4 + MAX_BANDS, 0);
    table
}

fn pages_buffer(device: &wgpu::Device, height: u32, prefix: u32, entries: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Shared atlas pages"),
        contents: bytemuck::cast_slice(&page_table(height, prefix, entries)),
        usage: wgpu::BufferUsages::UNIFORM,
    })
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
        // COPY_SRC: the pools are copied into a taller pool and into the whole atlas, and a
        // shader pack's weather copies the rain and snow sprites out of the whole atlas.
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
#[path = "shared_atlas/tests.rs"]
mod tests;
