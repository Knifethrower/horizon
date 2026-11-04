//! ASTC 4x4 for the textures Horizon assembles at run time.
//!
//! The shared atlas and the block sprite arrays are packed on the CPU from the
//! resource pack; the GPU only samples them afterwards. Compressed, they take a
//! quarter of the memory and a quarter of the bytes per sample, which is what a
//! bandwidth-limited handheld notices: ASTC 4x4 is 8 bits per texel against
//! `Rgba8Unorm`'s 32, and four times as many texels share a cache line.
//!
//! Compressing is a one-time job: `--bake-atlas` writes `cache/*.astc` beside
//! the RGBA spills `patch.bash` already leaves there, and the renderer loads
//! that file when this GPU samples ASTC. When it does not, or the file is
//! missing or the wrong size, every caller falls back to the RGBA path.

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The block a baked image is divided into.
pub const BLOCK: u32 = 4;
/// Bytes one block occupies in memory and on disk.
pub const BLOCK_BYTES: u64 = 16;
/// The 16-byte header `astcenc` puts in front of the blocks in a `.astc` file
/// (magic `0x5CA1AB13`, block dimensions, image dimensions). wgpu uploads raw
/// blocks, so a cache keeps only what follows it.
const ASTC_MAGIC: [u8; 4] = [0x13, 0xAB, 0xA1, 0x5C];

/// Bytes an ASTC 4x4 image of this size takes, blocks at the edges included.
pub fn encoded_len(width: u32, height: u32) -> u64 {
    u64::from(width.div_ceil(BLOCK)) * u64::from(height.div_ceil(BLOCK)) * BLOCK_BYTES
}

/// The format baked images upload as. Horizon's textures are linear
/// (`Rgba8Unorm`), so the channel is `Unorm`, never `UnormSrgb`.
pub fn format() -> wgpu::TextureFormat {
    wgpu::TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    }
}

/// Whether this device can sample ASTC. The GL backend reports the feature
/// from `GL_KHR_texture_compression_astc_ldr`, so this stays a capability
/// check: no device or SoC names anywhere.
pub fn supported(device: &wgpu::Device) -> bool {
    device
        .features()
        .contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
}

/// The bake of a shared atlas's static rows (`HudAtlas::static_rows`): its
/// width, those rows and their content key (`HudAtlas::static_key`), so a bake
/// of other textures of the same size is never taken for this one.
pub fn atlas_path(dir: &Path, width: u32, rows: u32, key: u64) -> PathBuf {
    dir.join(format!("{ATLAS_PREFIX}{width}x{rows}-{key:016x}.astc"))
}
/// The start of every shared atlas bake's name, old ones included.
pub const ATLAS_PREFIX: &str = "hud-atlas-";

/// The encoder to run: `tools/astcenc` beside the executable, or
/// `HORIZON_ASTCENC`. `None` leaves the RGBA path in place.
pub fn find_encoder() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("HORIZON_ASTCENC") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    ["tools/astcenc", "astcenc"]
        .into_iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

/// Upload the baked image at `path` as a texture, or `None` when this GPU
/// does not sample ASTC, or the file is missing, the wrong size for this
/// image, or unreadable. The file holds raw blocks (a bake from an older build
/// may still carry `astcenc`'s 16-byte container header, which is skipped).
/// Streams a band of block rows at a time so the transient copy stays a
/// megabyte or two, as the RGBA path does.
pub fn load(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &Path,
    width: u32,
    height: u32,
    label: &str,
) -> Option<wgpu::Texture> {
    let want = encoded_len(width, height);
    if !supported(device) {
        return None;
    }
    let mut file = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let len = std::fs::metadata(path).ok()?.len();
    if len == want + 16 {
        // A bake from a build that kept `astcenc`'s container header: the
        // blocks are the same, they just start sixteen bytes in.
        let mut head = [0u8; 16];
        std::io::Read::read_exact(&mut file, &mut head).ok()?;
        if head[..4] != ASTC_MAGIC {
            return None;
        }
    } else if len != want {
        return None;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format(),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // A block row of a 4096-wide image is 16 KiB; sixteen of them is a
    // quarter-megabyte, so even the largest atlas streams in a handful of
    // writes without a full copy in memory.
    let bytes_per_row = u64::from(width.div_ceil(BLOCK)) * BLOCK_BYTES;
    let block_rows = (1 << 20) / bytes_per_row.max(1);
    let rows_per_band = u32::try_from(block_rows).unwrap_or(1).max(1);
    let mut band = Vec::new();
    let mut block_row = 0u32;
    let total_rows = height.div_ceil(BLOCK);
    while block_row < total_rows {
        let rows = rows_per_band.min(total_rows - block_row);
        let len = (u64::from(rows) * bytes_per_row) as usize;
        band.resize(len, 0);
        std::io::Read::read_exact(&mut file, &mut band).ok()?;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: block_row * BLOCK,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &band,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row as u32),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width,
                height: rows * BLOCK,
                depth_or_array_layers: 1,
            },
        );
        block_row += rows;
    }
    Some(texture)
}

/// Write `raw` (`Rgba8Unorm`, top row first) as the TGA the encoder reads.
/// ASTC's encoder takes image files, not the raw plane the game keeps; a TGA
/// carries RGBA behind an 18-byte header. Streamed a band at a time, so an
/// 80 MiB atlas never costs a second copy of itself to convert.
pub fn write_tga(raw: &[u8], width: u32, height: u32, path: &Path) -> std::io::Result<()> {
    // A TGA header stores both dimensions in 16 bits; a taller image would
    // be written truncated and the encoder would read the wrong picture.
    if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{width}x{height} does not fit a TGA header"),
        ));
    }
    let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
    let mut header = [0u8; 18];
    header[2] = 2; // uncompressed true colour, no colour map
    header[12..14].copy_from_slice(&(width as u16).to_le_bytes());
    header[14..16].copy_from_slice(&(height as u16).to_le_bytes());
    header[16] = 32; // bits per pixel
    header[17] = 0x28; // 8 alpha bits, top row first
    out.write_all(&header)?;
    const BAND: usize = 1 << 20; // a multiple of four
    let mut band = vec![0u8; BAND];
    for chunk in raw.chunks(BAND) {
        let len = chunk.len() & !3;
        band[..len].copy_from_slice(&chunk[..len]);
        // TGA stores BGRA; the atlas is RGBA.
        for texel in band[..len].chunks_exact_mut(4) {
            texel.swap(0, 2);
        }
        out.write_all(&band[..len])?;
    }
    out.flush()
}

/// Compress `raw` to `out` with the encoder at `encoder`. Writes through a
/// temporary file and renames, so a half-written cache is never loaded.
pub fn encode(
    encoder: &Path,
    raw: &[u8],
    width: u32,
    height: u32,
    out: &Path,
) -> Result<(), String> {
    let tga = out.with_extension("tga");
    write_tga(raw, width, height, &tga).map_err(|e| format!("{}: {e}", tga.display()))?;
    // The encoder picks the output format from the file name, and `.astc.tmp`
    // reads as an unknown type ("Unknown compressed output file type '.tmp'"):
    // the temporary keeps `.astc` as its extension and a `tmp` step before it.
    let tmp = out.with_extension("tmp.astc");
    let status = std::process::Command::new(encoder)
        .arg("-cl")
        .arg(&tga)
        .arg(&tmp)
        .arg("4x4")
        .arg("-medium")
        .status()
        .map_err(|e| format!("could not run {}: {e}", encoder.display()))?;
    let _ = std::fs::remove_file(&tga);
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{} exited with {status}", encoder.display()));
    }
    // Keep the blocks and drop `astcenc`'s own container header: `load` checks
    // the cache against `encoded_len`, which counts blocks.
    let mut raw = std::io::BufReader::new(
        std::fs::File::open(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?,
    );
    let mut head = [0u8; 16];
    let headed = std::io::Read::read_exact(&mut raw, &mut head).is_ok() && head[..4] == ASTC_MAGIC;
    if !headed {
        std::io::Seek::seek(&mut raw, std::io::SeekFrom::Start(0))
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
    }
    let mut cache = std::io::BufWriter::new(
        std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?,
    );
    std::io::copy(&mut raw, &mut cache).map_err(|e| format!("{}: {e}", out.display()))?;
    std::io::Write::flush(&mut cache).map_err(|e| format!("{}: {e}", out.display()))?;
    drop(cache);
    let _ = std::fs::remove_file(&tmp);
    Ok(())
}

/// The key of packed sprite arrays, over their top-level pixels only.
///
/// Every other level is derived from those pixels, so a chain built with fewer
/// mip levels hashes like one built with more: the player's Mipmap Levels
/// option decides how many levels a run builds, and must not decide which bake
/// it can use. `chains` is `(layer-major chain, its level count)`, one per
/// array, in layer order.
pub fn pack_key<'a>(size: u32, chains: impl IntoIterator<Item = (&'a [u8], u32)>) -> u64 {
    let top = (size * size * 4) as usize;
    let mut parts: Vec<&[u8]> = Vec::new();
    for (bytes, levels) in chains {
        let chain: usize = (0..levels)
            .map(|level| {
                let edge = (size >> level).max(1);
                (edge * edge * 4) as usize
            })
            .sum();
        if chain == 0 || bytes.len() < top {
            continue;
        }
        for layer in 0..bytes.len() / chain {
            let start = layer * chain;
            parts.push(&bytes[start..start + top]);
        }
    }
    content_key_parts(parts)
}

/// A cache key for a packed array: the same bytes always give the same key,
/// so a cache baked from other resources is never mistaken for this one.
pub fn content_key(bytes: &[u8]) -> u64 {
    content_key_parts(std::iter::once(bytes))
}

/// [`content_key`] of several slices read as one. A pack cut into arrays for a
/// smaller adapter hashes like the one chain the bake had, because a chain is a
/// whole number of eight-byte words per layer.
pub fn content_key_parts<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut word = |bytes: &[u8]| {
        for chunk in bytes.chunks_exact(8) {
            hash ^= u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]));
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for &byte in bytes.chunks_exact(8).remainder() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for part in parts {
        word(part);
    }
    hash
}

/// The most mip levels a baked array of `size` pixels can hold: the chain down
/// to a single block, which is also the narrowest level a texture copy and an
/// ASTC texture can express. The bake builds this many whatever the Mipmap
/// Levels option says, so one bake serves every value of it.
pub fn max_levels(size: u32) -> u32 {
    size.ilog2().saturating_sub(BLOCK.ilog2()) + 1
}

/// The baked file of one packed sprite array.
/// The bake has no adapter, so it writes every layer into one file; the
/// renderer reads each of its arrays out of that file by layer range.
pub fn array_path(dir: &Path, size: u32, layers: u32, levels: u32, key: u64) -> PathBuf {
    dir.join(format!(
        "sprites-{size}-{layers}l-{levels}m-{key:016x}.astc"
    ))
}

/// The bake of this pack in `dir`, of any level count. A mip chain is a
/// prefix, so a bake with more levels than wanted serves a request for fewer:
/// the level count is the player's Mipmap Levels option and changes between
/// bakes and runs, which must not decide whether a bake is found at all.
/// The most levels wins when several match.
pub fn find_array(dir: &Path, size: u32, layers: u32, key: u64) -> Option<PathBuf> {
    let prefix = format!("sprites-{size}-{layers}l-");
    let suffix = format!("-{key:016x}.astc");
    let mut best: Option<(u32, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name();
        let Some(rest) = name.to_str().and_then(|name| name.strip_prefix(&prefix)) else {
            continue;
        };
        let Some(levels) = rest
            .strip_suffix(&suffix)
            .and_then(|levels| levels.strip_suffix('m'))
            .and_then(|levels| levels.parse::<u32>().ok())
        else {
            continue;
        };
        if best.as_ref().is_none_or(|(have, _)| levels > *have) {
            best = Some((levels, entry.path()));
        }
    }
    best.map(|(_, path)| path)
}

/// Bytes a packed array's baked file takes: a 16-byte header (`HZS1`, edge,
/// layers, levels) and every level's blocks, layers one after another.
pub fn array_len(size: u32, layers: u32, levels: u32) -> u64 {
    let mut total = 16;
    for level in 0..levels {
        let edge = (size >> level).max(1);
        total += encoded_len(edge, edge) * u64::from(layers);
    }
    total
}

/// Upload a baked sprite array, or `None` when the file is missing, of the
/// wrong size, or this device cannot sample ASTC. `layers` is what the file
/// holds; a texture with more layers gets zeroed ones (the packed arrays pad
/// their layer count for OpenGL).
pub fn load_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &Path,
    size: u32,
    first_layer: u32,
    layers: u32,
    texture_layers: u32,
    levels: u32,
    label: &str,
) -> Option<wgpu::Texture> {
    if !supported(device) {
        return None;
    }
    let mut file = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let mut header = [0u8; 16];
    std::io::Read::read_exact(&mut file, &mut header).ok()?;
    let u32_at = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().unwrap_or([0; 4]));
    let total = u32_at(8);
    let baked = u32_at(12);
    // More levels than asked for is fine: the chain is a prefix, and the
    // player's Mipmap Levels option picks how many of them are sampled.
    if &header[0..4] != b"HZS1" || u32_at(4) != size || baked < levels {
        return None;
    }
    // wgpu copies whole blocks: a level narrower than one block cannot upload.
    if (size >> (levels - 1)).max(1) < BLOCK {
        return None;
    }
    if layers > texture_layers
        || first_layer.checked_add(layers)? > total
        || std::fs::metadata(path).ok()?.len() != array_len(size, total, baked)
    {
        return None;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: texture_layers,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format(),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for level in 0..levels {
        let edge = (size >> level).max(1);
        let blocks = edge.div_ceil(BLOCK);
        let row_bytes = blocks * BLOCK_BYTES as u32;
        // A level holds every layer, one after another, so this array's range
        // is a seek in and a seek out.
        let layer_bytes = u64::from(blocks) * u64::from(row_bytes);
        let skip_in = u64::from(first_layer) * layer_bytes;
        let skip_out = u64::from(total - first_layer - layers) * layer_bytes;
        if std::io::Seek::seek(&mut file, std::io::SeekFrom::Current(skip_in as i64)).is_err() {
            return None;
        }
        let len = layer_bytes * u64::from(layers);
        let mut data = vec![0u8; len as usize];
        std::io::Read::read_exact(&mut file, &mut data).ok()?;
        // `write_texture` wants a 256-byte row pitch; the narrow levels pad
        // theirs, and each layer still holds `blocks` rows.
        let pitch = row_bytes.next_multiple_of(256);
        let padded;
        let (bytes, pitch) = if pitch == row_bytes {
            (&data[..], row_bytes)
        } else {
            let mut out = vec![0u8; (pitch * blocks * layers) as usize];
            for layer in 0..layers as usize {
                for row in 0..blocks as usize {
                    let src = ((layer * blocks as usize + row) * row_bytes as usize) as usize;
                    let dst = ((layer * blocks as usize + row) * pitch as usize) as usize;
                    out[dst..dst + row_bytes as usize]
                        .copy_from_slice(&data[src..src + row_bytes as usize]);
                }
            }
            padded = out;
            (&padded[..], pitch)
        };
        let write = |texture: &wgpu::Texture, origin_z: u32, layers: u32, bytes: &[u8], pitch: u32| {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: origin_z,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(blocks),
                },
                wgpu::Extent3d {
                    width: edge,
                    height: edge,
                    depth_or_array_layers: layers,
                },
            );
        };
        if std::io::Seek::seek(&mut file, std::io::SeekFrom::Current(skip_out as i64)).is_err() {
            return None;
        }
        write(&texture, 0, layers, bytes, pitch);
        let extra = texture_layers - layers;
        if extra > 0 {
            // Never sampled: the packed layer count OpenGL pads.
            let zero = vec![0u8; (pitch * blocks) as usize];
            for layer in 0..extra {
                write(&texture, layers + layer, 1, &zero, pitch);
            }
        }
    }
    Some(texture)
}

/// Compress a packed sprite array (`PreparedTextures`' layer-major mip chain)
/// into the file `load_array` reads. One tall image per level, so a 1291-layer
/// pack costs four encoder runs per array rather than one per layer.
pub fn encode_array(
    encoder: &Path,
    bytes: &[u8],
    size: u32,
    layers: u32,
    levels: u32,
    out: &Path,
) -> Result<(), String> {
    let chain: usize = (0..levels)
        .map(|level| {
            let edge = (size >> level).max(1) as usize;
            edge * edge * 4
        })
        .sum();
    if bytes.len() != chain * layers as usize {
        return Err(format!(
            "{}: {} bytes for {layers} layers of {size}px at {levels} levels",
            out.display(),
            bytes.len()
        ));
    }
    let tmp = out.with_extension("tmp.astc");
    {
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?,
        );
        let mut head = Vec::with_capacity(16);
        head.extend_from_slice(b"HZS1");
        head.extend_from_slice(&size.to_le_bytes());
        head.extend_from_slice(&layers.to_le_bytes());
        head.extend_from_slice(&levels.to_le_bytes());
        file.write_all(&head)
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
        let tga = out.with_extension("tga");
        let level_astc = out.with_extension("level.astc");
        for level in 0..levels {
            let edge = (size >> level).max(1);
            let layer_len = (edge * edge * 4) as usize;
            let offset: usize = (0..level)
                .map(|level| {
                    let edge = (size >> level).max(1) as usize;
                    edge * edge * 4
                })
                .sum();
            // A level is one tall image, `edge` wide and `edge * layers` high,
            // and a TGA's header holds 16-bit dimensions: 1291 layers of 64 px
            // is 82,624, past the 65,535 it can say, and the file went out
            // truncated (the encoder saw 17,088 px). Encode in runs that fit;
            // their blocks concatenate, a layer being `encoded_len` bytes in
            // every run.
            let per_run = (u16::MAX as u32 / edge).max(1) as usize;
            let mut first = 0;
            while first < layers as usize {
                let count = per_run.min(layers as usize - first);
                let mut image = Vec::with_capacity(layer_len * count);
                for layer in first..first + count {
                    let at = layer * chain + offset;
                    image.extend_from_slice(&bytes[at..at + layer_len]);
                }
                write_tga(&image, edge, edge * count as u32, &tga)
                    .map_err(|e| format!("{}: {e}", tga.display()))?;
                drop(image);
                let status = std::process::Command::new(encoder)
                    .arg("-cl")
                    .arg(&tga)
                    .arg(&level_astc)
                    .arg("4x4")
                    .arg("-medium")
                    .status()
                    .map_err(|e| format!("could not run {}: {e}", encoder.display()))?;
                if !status.success() {
                    let _ = std::fs::remove_file(&tga);
                    let _ = std::fs::remove_file(&level_astc);
                    let _ = std::fs::remove_file(&tmp);
                    return Err(format!("{} exited with {status}", encoder.display()));
                }
                let mut blocks = std::fs::read(&level_astc)
                    .map_err(|e| format!("{}: {e}", level_astc.display()))?;
                // `astcenc` puts its container header in front of the blocks;
                // the container this writes holds blocks only.
                if blocks.len() >= 16 && blocks[..4] == ASTC_MAGIC {
                    blocks.drain(..16);
                }
                let want = encoded_len(edge, edge) * count as u64;
                if blocks.len() as u64 != want {
                    let _ = std::fs::remove_file(&tga);
                    let _ = std::fs::remove_file(&level_astc);
                    let _ = std::fs::remove_file(&tmp);
                    return Err(format!(
                        "{}: {} bytes of blocks, expected {want} for {count} layers",
                        level_astc.display(),
                        blocks.len()
                    ));
                }
                file.write_all(&blocks)
                    .map_err(|e| format!("{}: {e}", tmp.display()))?;
                first += count;
            }
        }
        let _ = std::fs::remove_file(&tga);
        let _ = std::fs::remove_file(&level_astc);
        file.flush().map_err(|e| format!("{}: {e}", tmp.display()))?;
    }
    std::fs::rename(&tmp, out).map_err(|e| format!("{}: {e}", out.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bake of a shared atlas's static rows is named by their size and
    /// content key: another pack's atlas of the same size never matches it.
    #[test]
    fn an_atlas_bake_is_named_by_its_content() {
        let dir = Path::new("cache");
        let path = atlas_path(dir, 4096, 2608, 0xabc);
        assert_eq!(
            path,
            Path::new("cache/hud-atlas-4096x2608-0000000000000abc.astc")
        );
        assert_ne!(path, atlas_path(dir, 4096, 2608, 0xabd));
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(ATLAS_PREFIX)
        );
    }
}
