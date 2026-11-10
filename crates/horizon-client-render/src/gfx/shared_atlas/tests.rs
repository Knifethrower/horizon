use super::*;

/// A band that holds a written row is written, and only a written band holds pixels for the
/// dynamic pool; the last band may be short.
#[test]
fn bands_say_what_is_written_and_what_holds_pixels() {
    let (width, height) = (4, 5 * BAND_ROWS + 10);
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let texel = |x: u32, y: u32| ((y * width + x) * 4) as usize;
    // A fixed image in band 0, a written one with pixels in band 1, written rows in bands 4
    // and 5 (the short one) that hold nothing yet.
    pixels[texel(1, 3)] = 9;
    pixels[texel(2, BAND_ROWS + 5)] = 7;
    let written = [(BAND_ROWS + 2, BAND_ROWS + 9), (4 * BAND_ROWS + 60, height)];
    let bands = Bands::new(width, height, &written, &pixels);
    assert_eq!(bands.counts(), (3, 3, 1));
    let (entries, fixed, held) = bands.entries();
    assert_eq!((fixed, held), (3, 1));
    assert_eq!(entries, [1, DYNAMIC | 1, 2, 3, 0, 0]);
    // Nothing written: every band static, in place.
    let all = Bands::all_static(width, height);
    assert_eq!(all.counts(), (6, 0, 0));
    assert_eq!(prefix_rows(height, &all.entries().0), height);
    assert_eq!(prefix_rows(height, &entries), BAND_ROWS);
}

/// Rows are cut where a band ends.
#[test]
fn rows_are_cut_at_the_bands() {
    let b = BAND_ROWS;
    let cut = |y, rows| pieces(y, rows).collect::<Vec<_>>();
    assert_eq!(cut(0, 10), [(0, 0, 10)]);
    assert_eq!(cut(b - 4, 10), [(0, 0, 4), (1, 4, 6)]);
    assert_eq!(cut(b, 2 * b + 1), [(1, 0, b), (2, b, b), (3, 2 * b, 1)]);
    assert!(cut(5, 0).is_empty());
}

/// Rows go to where their bands are held, as one copy while the bands follow each other in one
/// pool, and nowhere for a band not held.
#[test]
fn rows_go_where_their_bands_are_held() {
    let b = BAND_ROWS;
    // Band 0 static at 0, band 1 written at the dynamic pool's 1, band 2 at its 0 (held first),
    // band 3 not held, band 4 static at 1.
    let entries = [1, DYNAMIC | 2, DYNAMIC | 1, 0, 2];
    assert_eq!(spans(&entries, 3, 5), [(false, 3, 0, 5)]);
    // Band 1 is held after band 2: two copies.
    assert_eq!(
        spans(&entries, b + 60, 8),
        [(true, b + 60, 0, 4), (true, 0, 4, 4)]
    );
    // Band 3 is not held; band 4 is.
    assert_eq!(spans(&entries, 3 * b + 1, b), [(false, b, b - 1, 1)]);
    assert!(spans(&entries, 3 * b, 4).is_empty());
    // The held bands in runs: band 0 alone, 1 and 2 apart in the pool, 4 after a gap.
    assert_eq!(
        runs(5 * b, &entries),
        [
            (false, 0, 0, b),
            (true, b, b, 2 * b),
            (true, 0, 2 * b, 3 * b),
            (false, b, 4 * b, 5 * b)
        ]
    );
}

/// The page table has its header and an entry for every band the shader can address, and the
/// shader's declarations are where the renderer binds them.
#[test]
fn the_shader_reads_the_page_table_the_renderer_writes() {
    let table = page_table(5120, 2304, &[1, 2, DYNAMIC | 1, 0]);
    assert_eq!(table.len() * 4, PAGES_BYTES as usize);
    assert_eq!(table[..8], [5120, 2304, 0, 0, 1, 2, DYNAMIC | 1, 0]);
    assert!(table[8..].iter().all(|&entry| entry == 0));
    let source = "@group(0) @binding(1) var atlas: texture_2d<f32>;\n".to_owned() + WGSL;
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let binding = |name: &str| {
        let (_, var) = module
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some(name))
            .unwrap();
        (var.binding.as_ref().map(|b| (b.group, b.binding)), var.ty)
    };
    assert_eq!(binding("atlas_dynamic").0, Some((0, DYNAMIC_BINDING)));
    let (pages, ty) = binding("atlas_pages");
    assert_eq!(pages, Some((0, PAGES_BINDING)));
    assert_eq!(
        u64::from(module.types[ty].inner.size(module.to_ctx())),
        PAGES_BYTES
    );
    let (_, rows) = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some("ATLAS_BAND_ROWS"))
        .unwrap();
    assert!(matches!(
        module.global_expressions[rows.init],
        naga::Expression::Literal(naga::Literal::U32(BAND_ROWS))
    ));
    assert!(WGSL.contains(&format!("/ ATLAS_BAND_ROWS, {}u)", MAX_BANDS - 1)));
}

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = crate::gfx::test_gpu::instance();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    pollster::block_on(adapter.request_device(&Default::default())).ok()
}

/// `texture`'s texels, tightly packed.
fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let padded = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(padded * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        copy_target(texture, 0, 0),
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

/// A 16-wide atlas of a static band, a written band with an animated sprite, a fixed image in
/// a written band, written bands that hold nothing, and a short static band at the bottom; and
/// its rows written at run time.
fn atlas_pixels() -> (u32, u32, Vec<u8>, Vec<(u32, u32)>) {
    let (width, height) = (16, 5 * BAND_ROWS + 20);
    let pixels: Vec<u8> = (0..width * height)
        .flat_map(|i| {
            let (x, y) = (i % width, i / width);
            let band = y / BAND_ROWS;
            // Bands 2 and 3 hold nothing yet; band 1 only from column 8 (a sprite beside a slot).
            if band == 2 || band == 3 || (band == 1 && x < 8) {
                [0; 4]
            } else {
                [x as u8 * 13, y as u8, (y >> 8) as u8 * 50 + 7, 255]
            }
        })
        .collect();
    let written = vec![(BAND_ROWS, 4 * BAND_ROWS - 3)];
    (width, height, pixels, written)
}

fn split_atlas(device: &wgpu::Device, queue: &wgpu::Queue) -> (SharedAtlas, Vec<u8>, u32, u32) {
    let (width, height, pixels, written) = atlas_pixels();
    let source = Source {
        pixels: &pixels,
        width,
        height,
        bands: Cow::Owned(Bands::new(width, height, &written, &pixels)),
    };
    let atlas = SharedAtlas::new(device, queue, &source, None).unwrap();
    (atlas, pixels, width, height)
}

/// The written bands are held only once something is written into them, the pool grows as
/// writes reach new ones, and the atlas reads back as written wherever its bands are held; a
/// band never written reads as transparent black.
#[test]
fn the_dynamic_pool_holds_only_the_bands_written() {
    let Some((device, queue)) = device() else {
        eprintln!("no adapter: skipped");
        return;
    };
    let (atlas, mut pixels, width, _) = split_atlas(&device, &queue);
    // Band 1 has pixels; bands 2 and 3 not yet.
    assert_eq!(atlas.held_bands(), (1, 3));
    // Bands 0, 4 and the short band 5 are static.
    let band = u64::from(width * BAND_ROWS * 4);
    assert_eq!(atlas.bytes(), (3 * band, band));
    let generation = atlas.generation();
    let mut uploads = Uploads::new(&device);
    let mut encoder = device.create_command_encoder(&Default::default());
    // Transparent black into band 2 holds nothing; a sprite across bands 2 and 3 holds both,
    // after band 1.
    atlas.write(&mut uploads, &mut encoder, 0, 2 * BAND_ROWS, 4, 4, &[0; 64]);
    assert_eq!(atlas.held_bands(), (1, 3));
    let (x, y, w, h) = (3, 3 * BAND_ROWS - 2, 2, 4);
    let patch: Vec<u8> = (0..w * h * 4).map(|i| i as u8 + 1).collect();
    atlas.write(&mut uploads, &mut encoder, x, y, w, h, &patch);
    uploads.finish(&encoder);
    assert_eq!(atlas.held_bands(), (3, 3));
    assert!(atlas.generation() > generation);
    for row in 0..h {
        let at = ((y + row) * width + x) as usize * 4;
        pixels[at..at + (w * 4) as usize]
            .copy_from_slice(&patch[(row * w * 4) as usize..((row + 1) * w * 4) as usize]);
    }
    let whole = atlas.whole(&device, &queue, &mut encoder);
    assert!(read_back(&device, &queue, encoder, &whole) == pixels);
    // From now on the whole atlas follows the writes too.
    let mut encoder = device.create_command_encoder(&Default::default());
    atlas.write(&mut uploads, &mut encoder, 9, 5, 1, 1, &[1, 2, 3, 4]);
    uploads.finish(&encoder);
    pixels[((5 * width + 9) * 4) as usize..][..4].copy_from_slice(&[1, 2, 3, 4]);
    let whole = atlas.whole(&device, &queue, &mut encoder);
    assert!(read_back(&device, &queue, encoder, &whole) == pixels);
}

/// What `shared_atlas.wgsl` reads at every texel of the atlas, through `atlas_sample` with a
/// nearest sampler, `atlas_load`, and `atlas_linear` a quarter texel down with a linear one.
fn draw_reads(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    parts: &Parts,
    (width, height): (u32, u32),
) -> [Vec<u8>; 3] {
    let source = with(
        "@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4(f32(i % 2u) * 4.0 - 1.0, f32(i / 2u) * 4.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs_sample(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return atlas_sample(atlas_sampler, p.xy / vec2<f32>(atlas_size()));
}
@fragment fn fs_load(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return atlas_load(vec2<i32>(p.xy));
}
@fragment fn fs_linear(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return atlas_linear(atlas_sampler, (p.xy + vec2(0.0, 0.25)) / vec2<f32>(atlas_size()));
}
",
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let nearest = device.create_sampler(&Default::default());
    let linear = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    ["fs_sample", "fs_load", "fs_linear"].map(|entry| {
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(view),
        }];
        if entry != "fs_load" {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(if entry == "fs_linear" {
                    &linear
                } else {
                    &nearest
                }),
            });
        }
        entries.extend(parts.entries());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: &[],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let view = target.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        read_back(device, queue, encoder, &target)
    })
}

/// A draw reads the split atlas texel for texel as it reads one texture holding the same
/// pixels, the bands never written as transparent black; and a linear sample across a band
/// edge where the two bands are held apart blends the same two rows.
#[test]
fn a_draw_reads_the_split_atlas_as_one_texture() {
    let Some((device, queue)) = device() else {
        eprintln!("no adapter: skipped");
        return;
    };
    let (atlas, pixels, width, height) = split_atlas(&device, &queue);
    // Band 3 held before band 2, so they are apart in the pool.
    let mut uploads = Uploads::new(&device);
    let mut encoder = device.create_command_encoder(&Default::default());
    let row = vec![200u8; (width * 4) as usize];
    let mut written = pixels.clone();
    for y in [
        3 * BAND_ROWS,
        3 * BAND_ROWS + 1,
        3 * BAND_ROWS - 1,
        2 * BAND_ROWS,
    ] {
        atlas.write(&mut uploads, &mut encoder, 0, y, width, 1, &row);
        written[(y * width * 4) as usize..][..row.len()].copy_from_slice(&row);
    }
    uploads.finish(&encoder);
    queue.submit([encoder.finish()]);
    let plain = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &written,
    );
    let split = draw_reads(
        &device,
        &queue,
        atlas.view(),
        &atlas.parts(),
        (width, height),
    );
    let one = draw_reads(
        &device,
        &queue,
        &plain.create_view(&Default::default()),
        &Parts::whole(&device),
        (width, height),
    );
    assert!(
        split[0] == written,
        "atlas_sample differs from the atlas's texels"
    );
    assert!(
        split[1] == written,
        "atlas_load differs from the atlas's texels"
    );
    assert!(one[0] == written && one[1] == written);
    let worst = split[2]
        .iter()
        .zip(&one[2])
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(worst <= 1, "atlas_linear differs by {worst}");
}

/// The static pool, as its texture and its bake hold it: the static bands in order, a short
/// one filled to a band; named by a key of their pixels alone.
#[test]
fn the_static_pool_is_the_static_bands_in_order() {
    let (width, height, mut pixels, written) = atlas_pixels();
    let bands = Bands::new(width, height, &written, &pixels);
    assert_eq!(bands.static_rows(), 3 * BAND_ROWS);
    let pool = bands.static_pool(&pixels);
    let row = (width * 4) as usize;
    let band = |b: u32, rows: u32| &pixels[(b * BAND_ROWS) as usize * row..][..rows as usize * row];
    let b = BAND_ROWS as usize * row;
    assert!(pool[..b] == *band(0, BAND_ROWS));
    assert!(pool[b..2 * b] == *band(4, BAND_ROWS));
    assert!(pool[2 * b..2 * b + 20 * row] == *band(5, 20));
    assert!(pool[2 * b + 20 * row..].iter().all(|&byte| byte == 0));
    // A written band's pixels name no bake; a static band's do.
    let key = bands.static_key();
    pixels[BAND_ROWS as usize * row + 40] ^= 1;
    assert_eq!(
        Bands::new(width, height, &written, &pixels).static_key(),
        key
    );
    pixels[4 * BAND_ROWS as usize * row + 40] ^= 1;
    assert_ne!(
        Bands::new(width, height, &written, &pixels).static_key(),
        key
    );
}

/// An ASTC 4x4 block of one colour (a void-extent block): `rgba` as 16-bit UNORM.
fn solid_block(rgba: [u16; 4]) -> [u8; 16] {
    let mut block = [0xff; 16];
    block[..2].copy_from_slice(&[0xfc, 0xfd]);
    for (i, c) in rgba.iter().enumerate() {
        block[8 + 2 * i..10 + 2 * i].copy_from_slice(&c.to_le_bytes());
    }
    block
}

/// With the spill and its bake there and a GPU that samples ASTC, the static bands are the
/// bake (what it holds is what draws read), the written ones come from the spill, the whole
/// atlas reads its static bands from the spill, and a write into a static band is dropped.
#[test]
fn the_static_bands_load_from_their_bake() {
    let instance = crate::gfx::test_gpu::instance();
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no adapter: skipped");
        return;
    };
    let astc = wgpu::Features::TEXTURE_COMPRESSION_ASTC;
    if !adapter.features().contains(astc) {
        eprintln!("the adapter samples no ASTC: skipped");
        return;
    }
    let descriptor = wgpu::DeviceDescriptor {
        required_features: astc,
        ..Default::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor)).unwrap();
    let (width, height, pixels, written) = atlas_pixels();
    let bands = Bands::new(width, height, &written, &pixels);
    let dir = std::env::temp_dir().join(format!("horizon-atlas-bake-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spill = dir.join("hud-atlas.rgba");
    std::fs::write(&spill, &pixels).unwrap();
    let rows = bands.static_rows();
    let bake = crate::astc::atlas_path(&dir, width, rows, bands.static_key());
    let blocks = (width / 4 * rows / 4) as usize;
    let solid = solid_block([0x8080, 0xffff, 0, 0xffff]);
    std::fs::write(&bake, solid.repeat(blocks)).unwrap();
    let source = Source {
        pixels: &[],
        width,
        height,
        bands: Cow::Owned(bands),
    };
    let atlas = SharedAtlas::new(&device, &queue, &source, Some(&spill)).unwrap();
    assert!(atlas.compressed());
    assert_eq!(atlas.bytes().0, crate::astc::encoded_len(width, rows));
    // A write into static band 0 is dropped.
    let mut uploads = Uploads::new(&device);
    let mut encoder = device.create_command_encoder(&Default::default());
    atlas.write(&mut uploads, &mut encoder, 2, 3, 1, 1, &[9, 9, 9, 9]);
    uploads.finish(&encoder);
    queue.submit([encoder.finish()]);
    let reads = draw_reads(
        &device,
        &queue,
        atlas.view(),
        &atlas.parts(),
        (width, height),
    );
    for y in 0..height {
        let band = y / BAND_ROWS;
        let at = (y * width * 4) as usize;
        let row = &reads[0][at..at + (width * 4) as usize];
        if band == 0 || band >= 4 {
            assert!(
                row.chunks(4).all(|texel| texel == [128, 255, 0, 255]),
                "row {y}"
            );
        } else {
            assert!(row == &pixels[at..at + (width * 4) as usize], "row {y}");
        }
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    let whole = atlas.whole(&device, &queue, &mut encoder);
    assert!(read_back(&device, &queue, encoder, &whole) == pixels);
    let _ = std::fs::remove_dir_all(&dir);
}
