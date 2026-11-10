//! The HUD drawn from the shared atlas held in bands (`gfx::shared_atlas`).
use super::*;
use crate::gfx::shared_atlas::{BAND_ROWS, Bands, SharedAtlas, Source};

/// `texture`'s texels, tightly packed.
fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let padded = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(padded * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    mapped
        .chunks(padded as usize)
        .flat_map(|row| row[..(width * 4) as usize].to_vec())
        .collect()
}

/// The HUD draws the atlas held in bands texel for texel as it draws the same pixels from one
/// texture of their own (a GUI picture, which binds its texture whole): a fixed band, a written
/// band with pixels from the start, one never written (transparent, so nothing is drawn) and
/// one a patch writes this frame.
#[test]
fn a_hud_draw_reads_the_split_atlas_as_one_texture() {
    let instance = crate::gfx::test_gpu::instance();
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no adapter: skipped");
        return;
    };
    let Ok((device, queue)) = pollster::block_on(adapter.request_device(&Default::default()))
    else {
        eprintln!("no device: skipped");
        return;
    };
    let (width, height) = (32, 4 * BAND_ROWS);
    let mut pixels: Vec<u8> = (0..width * height)
        .flat_map(|i| {
            let (x, y) = (i % width, i / width);
            if y >= 2 * BAND_ROWS {
                [0; 4]
            } else {
                [x as u8 * 8, y as u8, 99, 255]
            }
        })
        .collect();
    let bands = Bands::new(width, height, &[(BAND_ROWS, height)], &pixels);
    let source = Source {
        pixels: &pixels,
        width,
        height,
        bands: std::borrow::Cow::Owned(bands),
    };
    let atlas = SharedAtlas::new(&device, &queue, &source, None).unwrap();
    assert_eq!(atlas.held_bands(), (1, 3));
    let glint = device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut hud = HudRenderer::new(
        &device,
        format,
        &atlas,
        &glint,
        &glint,
        &device.create_sampler(&Default::default()),
        false,
    )
    .unwrap();
    // A patch into band 3, as a glyph is written.
    let patch = crate::font::Patch {
        x: 4,
        y: 3 * BAND_ROWS + 10,
        width: 8,
        height: 4,
        rgba: (0..8 * 4).flat_map(|i| [200, i as u8, 30, 255]).collect(),
    };
    for row in 0..patch.height {
        let at = (((patch.y + row) * width + patch.x) * 4) as usize;
        let from = (row * patch.width * 4) as usize;
        pixels[at..at + (patch.width * 4) as usize]
            .copy_from_slice(&patch.rgba[from..from + (patch.width * 4) as usize]);
    }
    let picture = Arc::new(pictures::Picture {
        id: 7,
        width,
        height,
        pixels: pixels.clone(),
        flags: TextureFlags::default(),
    });
    let texture = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let mut uploads = crate::gfx::upload::Uploads::new(&device);
    // The whole atlas over the target, from the atlas (with the patch) or from the picture.
    let mut draw = |from_picture: bool| {
        let mut batch = HudBatch::new([width, height]);
        let rect = Rect {
            u0: 0.0,
            v0: 0.0,
            u1: 1.0,
            v1: 1.0,
            w: width,
            h: height,
        };
        batch.rect_uv(0.0, 0.0, width as f32, height as f32, rect, u32::MAX);
        if from_picture {
            batch.pictures.push(picture.clone());
            batch.blend_since(0, Blend::Picture(picture.id));
        } else {
            batch.patches.push(crate::font::Patch {
                x: patch.x,
                y: patch.y,
                width: patch.width,
                height: patch.height,
                rgba: patch.rgba.clone(),
            });
        }
        let target = texture(
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = texture(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        let size = [width, height];
        hud.prepare(&device, &mut uploads, &mut encoder, &batch, size);
        let [target_view, depth_view] =
            [&target, &depth].map(|t| t.create_view(&Default::default()));
        hud.draw(&mut encoder, &target_view, &depth_view, &batch, size);
        uploads.finish(&encoder);
        queue.submit([encoder.finish()]);
        read_back(&device, &queue, &target)
    };
    let split = draw(false);
    assert_eq!(atlas.held_bands(), (2, 3), "the patch holds band 3");
    let one = draw(true);
    assert!(split == one, "the HUD reads the split atlas differently");
    // Each texel drawn as the atlas holds it, and the transparent band left as cleared.
    let at = |x: u32, y: u32| (((y * width) + x) * 4) as usize;
    for (x, y) in [(3, 5), (31, BAND_ROWS + 7), (5, 3 * BAND_ROWS + 12)] {
        assert_eq!(
            split[at(x, y)..at(x, y) + 4],
            pixels[at(x, y)..at(x, y) + 4]
        );
    }
    assert_eq!(split[at(9, 2 * BAND_ROWS + 5)..][..4], [0; 4]);
}
