//! The lean terrain shaders (`LeanTerrain`) against the ones they stand in for: one scene of
//! sections, drawn by the two solid passes through `vs_terrain_opaque`/`fs_opaque` and
//! `vs_terrain_cutout`/`fs_solid` as the renderer builds them, and through the lean pairs.
//!
//! They differ at a few pixels on the edges of faces, where a section's UV interpolates a hair
//! past 1: `sprite_at` takes that for a distant tile's next block (`tile_uv`) and samples the
//! other side of the sprite at the tile's single level, where the lean shaders, which draw
//! sections alone, take the sprite's own edge. So the full shaders are drawn a second time
//! without the wrap (`UNWRAPPED`), which the lean ones must match pixel for pixel, but for a
//! driver's rounding: at most 1 in a channel, in at most 1 pixel of 10,000.

use super::{Globals, TERRAIN_ATTRIBUTES, lean_frame, section_instance};
use crate::mesher::{OPAQUE_LAYER, TerrainVertex};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// terrain.wgsl's `tile_uv`, and in its place a version that leaves every UV as it is: the same
/// for UVs in [0, 1], as a section's are.
const UNWRAPPED: (&str, &str) = ("return uv - max(ceil(uv) - 1.0, vec2(0.0));", "return uv;");
/// A handheld's screen.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const SPRITE: u32 = 16;
const MIPS: u32 = 5;
/// Sprite layers: 0 opaque, 1 cut out, 2 graded alpha, 3 animated (into `animated`'s layer 1).
/// Neither count is 1 or a multiple of 6, which OpenGL would make a 2D texture or a cube map.
const LAYERS: u32 = 4;
const ANIMATED_LAYERS: u32 = 3;
const FOG: [f32; 4] = [0.62, 0.71, 0.83, 1.0];

struct Random(u32);
impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// A device, its queue and its backend.
fn gpu() -> Option<(wgpu::Device, wgpu::Queue, wgpu::Backend)> {
    let instance = super::test_gpu::instance();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    eprintln!("lean terrain GPU {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue, adapter.get_info().backend))
}

/// An RGBA8 array with its mip chain, each level its own noise; `alpha` picks a texel's alpha
/// from its layer and a random byte.
fn sprite_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layers: u32,
    seed: u32,
    alpha: impl Fn(u32, u32) -> u8,
) -> wgpu::TextureView {
    let mut random = Random(seed);
    let mut data = Vec::new();
    for layer in 0..layers {
        for level in 0..MIPS {
            let edge = SPRITE >> level;
            for _ in 0..edge * edge {
                let r = random.next();
                data.extend_from_slice(&[r as u8, (r >> 8) as u8, (r >> 16) as u8]);
                data.push(alpha(layer, random.next()));
            }
        }
    }
    device
        .create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("lean test sprites"),
                size: wgpu::Extent3d {
                    width: SPRITE,
                    height: SPRITE,
                    depth_or_array_layers: layers,
                },
                mip_level_count: MIPS,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &data,
        )
        .create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
}

fn texture_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    format: wgpu::TextureFormat,
    data: &[u8],
) -> wgpu::TextureView {
    let height = (data.len() as u32 / format.block_copy_size(None).unwrap()) / width;
    device
        .create_texture_with_data(
            queue,
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
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        )
        .create_view(&Default::default())
}

/// A field of block columns in front of the camera, in sections of 16 by 16 columns: each
/// column's top and the sides its lower neighbours expose, with every kind of sprite, smooth
/// light, a tint and alpha bits (the emission tier, which only HDR output reads). Each section's
/// vertices, its camera-relative origin and how far through vanilla's fade-in it is.
fn scene() -> Vec<(Vec<TerrainVertex>, [f32; 3], f32)> {
    let mut random = Random(0x9e37_79b9);
    let height = |x: i32, z: i32| -> i32 {
        let (x, z) = (x as f32, z as f32);
        (-6.0 + 2.5 * (x * 0.31).sin() + 2.0 * (z * 0.23 + x * 0.07).cos()).floor() as i32
    };
    let mut sections = Vec::new();
    for sx in -2..2 {
        for sz in -5..0 {
            let origin = [sx as f32 * 16.0, -16.0, sz as f32 * 16.0];
            let mut vertices = Vec::new();
            for bx in 0..16 {
                for bz in 0..16 {
                    let (x, z) = (sx * 16 + bx, sz * 16 + bz);
                    let top = height(x, z) + 1;
                    let mut face = |corners: [Vec3; 4], outward: Vec3| {
                        // (layer, marked without cutout): opaque texels, cut out, graded,
                        // animated, and leaves forced opaque over cut-out texels.
                        let kinds: [(u16, bool); 5] =
                            [(0, true), (1, false), (2, false), (3, false), (1, true)];
                        let (layer, whole) = kinds[random.below(kinds.len() as u32) as usize];
                        let mut corners = corners;
                        if (corners[1] - corners[0])
                            .cross(corners[2] - corners[0])
                            .dot(outward)
                            < 0.0
                        {
                            corners.reverse();
                        }
                        // Some faces take part of their sprite, as a slab's sides do.
                        let (u0, u1) = if random.below(3) == 0 {
                            (0.25, 0.75)
                        } else {
                            (0.0, 1.0)
                        };
                        let uv = [[u0, u0], [u0, u1], [u1, u1], [u1, u0]];
                        for (corner, uv) in corners.into_iter().zip(uv) {
                            let light = (random.below(16) * 16) | ((random.below(16) * 16) << 8);
                            let shade = 160 + random.below(96);
                            let color = shade
                                | (shade - random.below(40)) << 8
                                | (shade - random.below(60)) << 16
                                | random.below(256) << 24;
                            let p = corner - Vec3::from(origin);
                            vertices.push(TerrainVertex::new(
                                p.to_array(),
                                uv,
                                layer | if whole { OPAQUE_LAYER } else { 0 },
                                color,
                                light as u16,
                            ));
                        }
                    };
                    let (x0, x1) = (x as f32, x as f32 + 1.0);
                    let (z0, z1) = (z as f32, z as f32 + 1.0);
                    let y1 = top as f32;
                    face(
                        [
                            Vec3::new(x0, y1, z0),
                            Vec3::new(x0, y1, z1),
                            Vec3::new(x1, y1, z1),
                            Vec3::new(x1, y1, z0),
                        ],
                        Vec3::Y,
                    );
                    for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let below = height(x + dx, z + dz) + 1;
                        for y in below..top {
                            let (ya, yb) = (y as f32, y as f32 + 1.0);
                            let (xa, za, xb, zb) = match (dx, dz) {
                                (1, _) => (x1, z0, x1, z1),
                                (-1, _) => (x0, z0, x0, z1),
                                (_, 1) => (x0, z1, x1, z1),
                                _ => (x0, z0, x1, z0),
                            };
                            face(
                                [
                                    Vec3::new(xa, ya, za),
                                    Vec3::new(xb, ya, zb),
                                    Vec3::new(xb, yb, zb),
                                    Vec3::new(xa, yb, za),
                                ],
                                Vec3::new(dx as f32, 0.0, dz as f32),
                            );
                        }
                    }
                }
            }
            // Most sections whole, some fading in.
            let visibility = [1.0, 1.0, 0.3, 1.0, 0.75][sections.len() % 5];
            sections.push((vertices, origin, visibility));
        }
    }
    sections
}

struct Bed {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    /// terrain.wgsl with `tile_uv` left out (`UNWRAPPED`).
    unwrapped: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    bindings: wgpu::BindGroupLayout,
    sprites: wgpu::TextureView,
    lightmap: wgpu::TextureView,
    thresholds: wgpu::TextureView,
    layer_map: wgpu::TextureView,
    animated: wgpu::TextureView,
    sections: Vec<(wgpu::Buffer, u32, [f32; 4])>,
    indices: wgpu::Buffer,
}

/// What a frame changes: the Texture Filtering option, the fog, and the sprite arrays' split.
struct Case {
    filtering: u32,
    fog: [f32; 4],
    split: Option<u32>,
}

impl Bed {
    fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Horizon terrain"),
            source: wgpu::ShaderSource::Wgsl(super::terrain_shader().into()),
        });
        let source = super::terrain_shader();
        assert!(source.contains(UNWRAPPED.0), "terrain.wgsl's `tile_uv`");
        let unwrapped = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Horizon terrain, no tile wrap"),
            source: wgpu::ShaderSource::Wgsl(source.replace(UNWRAPPED.0, UNWRAPPED.1).into()),
        });
        // Renderer::new's terrain bindings.
        let entries: Vec<_> = (0..15)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: match binding {
                    0 => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    2 | 4 => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    3 | 12 | 13 => wgpu::BindingType::Texture {
                        sample_type: if binding == 13 {
                            wgpu::TextureSampleType::Uint
                        } else {
                            wgpu::TextureSampleType::Float { filterable: true }
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    _ => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                },
                count: None,
            })
            .collect();
        let bindings = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Horizon terrain bindings"),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bindings)],
            ..Default::default()
        });
        let sprites = sprite_array(&device, &queue, LAYERS, 7, |layer, r| match layer {
            1 => [0, 255][(r & 1) as usize],
            2 => r as u8,
            _ => 255,
        });
        let animated = sprite_array(&device, &queue, ANIMATED_LAYERS, 11, |_, r| {
            if r % 5 == 0 { 0 } else { 255 }
        });
        let mut random = Random(23);
        let lightmap: Vec<u8> = (0..256)
            .flat_map(|_| {
                let r = random.next();
                [r as u8, (r >> 8) as u8, (r >> 16) as u8, 255]
            })
            .collect();
        let lightmap = texture_2d(
            &device,
            &queue,
            16,
            wgpu::TextureFormat::Rgba8Unorm,
            &lightmap,
        );
        let thresholds = texture_2d(&device, &queue, 1, wgpu::TextureFormat::Rgba8Unorm, &[0; 4]);
        let mut map = [u32::MAX; LAYERS as usize];
        map[3] = 1;
        let layer_map = texture_2d(
            &device,
            &queue,
            LAYERS,
            wgpu::TextureFormat::R32Uint,
            bytemuck::cast_slice(&map),
        );
        let mut quads = 0;
        let sections = scene()
            .into_iter()
            .map(|(vertices, origin, visibility)| {
                quads = quads.max(vertices.len() as u32 / 4);
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                (
                    buffer,
                    vertices.len() as u32 / 4,
                    section_instance(origin, 0, visibility),
                )
            })
            .collect();
        let indices = super::Renderer::quad_indices(&device, quads);
        Self {
            device,
            queue,
            shader,
            unwrapped,
            layout,
            bindings,
            sprites,
            lightmap,
            thresholds,
            layer_map,
            animated,
            sections,
            indices,
        }
    }

    /// A solid pipeline as `Renderer::new` builds them without HDR output: back faces culled,
    /// reversed depth written, no blend, into the swapchain's 8 bits.
    fn pipeline(
        &self,
        module: &wgpu::ShaderModule,
        vertex: &str,
        fragment: &str,
        split: Option<u32>,
    ) -> wgpu::RenderPipeline {
        let instance_attributes = wgpu::vertex_attr_array![5=>Float32x4];
        let constants = [
            ("SPRITE_ARRAYS", if split.is_some() { 2.0 } else { 1.0 }),
            ("SPRITE_SPLIT", f64::from(split.unwrap_or(2048))),
            ("HDR_LIGHTING", 0.0),
            ("HDR_WRITE_LIGHT", 0.0),
        ];
        let options = wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        };
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&self.layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vertex),
                    compilation_options: options.clone(),
                    buffers: &[
                        Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<TerrainVertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &TERRAIN_ATTRIBUTES,
                        }),
                        Some(wgpu::VertexBufferLayout {
                            array_stride: 16,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &instance_attributes,
                        }),
                    ],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fragment),
                    compilation_options: options,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn globals(case: &Case) -> Globals {
        let mut globals: Globals = bytemuck::Zeroable::zeroed();
        let view = Mat4::look_to_rh(Vec3::ZERO, Vec3::new(0.15, -0.35, -1.0), Vec3::Y);
        let proj = Mat4::perspective_infinite_reverse_rh(
            70_f32.to_radians(),
            WIDTH as f32 / HEIGHT as f32,
            0.05,
        );
        globals.view_proj = (proj * view).to_cols_array_2d();
        globals.fog_color = FOG;
        globals.fog = case.fog;
        globals.filtering = [case.filtering as f32, SPRITE as f32, 0.0, 0.0];
        globals
    }

    /// The scene's two solid passes, each over every section (the other pass's quads left out,
    /// as for a mesh that was not sorted), with `passes`' (vertex, fragment) entries of `module`.
    /// RGBA8 rows.
    fn draw(&self, case: &Case, module: &wgpu::ShaderModule, passes: [(&str, &str); 2]) -> Vec<u8> {
        let globals = Self::globals(case);
        assert!(lean_frame(&globals, true), "a lean frame");
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&globals),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let anisotropy = if case.filtering == 2 { 16 } else { 1 };
        let sampler = self
            .device
            .create_sampler(&super::sprite_sampler_descriptor(anisotropy));
        let lightmap_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sprites = std::array::from_fn::<_, 8, _>(|_| self.sprites.clone());
        let group = super::terrain_group(
            &self.device,
            "lean test",
            &self.bindings,
            &uniform,
            &sprites,
            &sampler,
            &self.lightmap,
            &lightmap_sampler,
            &self.thresholds,
            &self.animated,
            &self.layer_map,
        );
        let target = |format, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                view_formats: &[],
            })
        };
        let color = target(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::COPY_SRC,
        );
        let depth = target(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::empty(),
        );
        let pipelines =
            passes.map(|(vertex, fragment)| self.pipeline(module, vertex, fragment, case.split));
        let instances: Vec<[f32; 4]> = self.sections.iter().map(|s| s.2).collect();
        let instances = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let color_view = color.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Alpha 0 where nothing is drawn: the solid passes write 1.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &group, &[]);
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            for pipeline in &pipelines {
                pass.set_pipeline(pipeline);
                for (i, (vertices, quads, _)) in self.sections.iter().enumerate() {
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    pass.set_vertex_buffer(1, instances.slice(i as u64 * 16..));
                    pass.draw_indexed(0..quads * 6, 0, 0..1);
                }
            }
        }
        let row = WIDTH * 4;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * HEIGHT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            color.size(),
        );
        self.queue.submit([encoder.finish()]);
        buffer.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        buffer.get_mapped_range(..).unwrap().to_vec()
    }
}

/// The pixels at which two of `draw`'s frames differ.
fn differing(a: &[u8], b: &[u8]) -> Vec<usize> {
    a.chunks(4)
        .zip(b.chunks(4))
        .enumerate()
        .filter_map(|(i, (p, q))| (p != q).then_some(i))
        .collect()
}

#[test]
fn lean_frames_draw_the_full_shaders_picture() {
    let Some((device, queue, backend)) = gpu() else {
        eprintln!("lean terrain: no GPU");
        return;
    };
    let bed = Bed::new(device, queue);
    let full = [
        ("vs_terrain_opaque", "fs_opaque"),
        ("vs_terrain_cutout", "fs_solid"),
    ];
    let lean = [
        ("vs_terrain_lean_opaque", "fs_lean_opaque"),
        ("vs_terrain_lean_cutout", "fs_lean_cutout"),
    ];
    let cases = [
        // No filtering; environmental fog and the render distance's.
        Case {
            filtering: 0,
            fog: [8.0, 48.0, 30.0, 60.0],
            split: None,
        },
        // No environmental fog (both f32::MAX, as `draw_frame` gives it), and the layers in two
        // arrays.
        Case {
            filtering: 0,
            fog: [f32::MAX, f32::MAX, 20.0, 50.0],
            split: Some(2),
        },
        // Environmental fog at one distance (a fog fading in from nothing), and one that ends
        // before it starts (a boss's fog nearer than the weather's start).
        Case {
            filtering: 0,
            fog: [15.0, 15.0, 40.0, 70.0],
            split: None,
        },
        Case {
            filtering: 0,
            fog: [30.0, 20.0, 40.0, 70.0],
            split: None,
        },
        // Anisotropic filtering.
        Case {
            filtering: 2,
            fog: [8.0, 48.0, 30.0, 60.0],
            split: None,
        },
    ];
    for (i, case) in cases.iter().enumerate() {
        let a = bed.draw(case, &bed.shader, full);
        let unwrapped = bed.draw(case, &bed.unwrapped, full);
        let b = bed.draw(case, &bed.shader, lean);
        let pixels = a.len() / 4;
        let drawn = a.chunks(4).filter(|p| p[3] == 255).count();
        let colours: std::collections::HashSet<_> = a.chunks(4).collect();
        assert!(drawn > pixels / 2, "case {i}: the scene covers the frame");
        assert!(
            colours.len() > 5000,
            "case {i}: sprites, light and fog vary"
        );
        // Where the wrap catches a section's UV, a few pixels on the edges of faces.
        let wrapped = differing(&a, &unwrapped);
        // Without it the full shaders draw the lean ones' picture, pixel for pixel but for
        // rounding: a driver compiles the two shaders' same arithmetic each in its own way
        // (llvmpipe's OpenGL moved 3 of 307,200 pixels by 1 in one channel, its Vulkan none).
        let rounded = differing(&unwrapped, &b);
        let most = rounded
            .iter()
            .flat_map(|&n| (n * 4..n * 4 + 4).map(|c| unwrapped[c].abs_diff(b[c])))
            .max()
            .unwrap_or(0);
        eprintln!(
            "case {i}: {drawn} of {pixels} pixels drawn, {} colours; {} differ from the full \
             shaders': {} where their tile wrap caught a section's UV, the rest of the frame the \
             same but {} pixels by at most {most}",
            colours.len(),
            differing(&a, &b).len(),
            wrapped.len(),
            rounded.len(),
        );
        // On llvmpipe's OpenGL, anisotropic, the full shaders' picture and their own without the
        // wrap differ in about 3% of its pixels (far cut-out texels one keeps and the other
        // discards), though the wrap fires at only a few: there the same anisotropic sample
        // doesn't come out the same in two programs, so the case shows nothing. Its Vulkan
        // compares this case as the others.
        if case.filtering == 2 && backend == wgpu::Backend::Gl {
            eprintln!("case {i}: not compared on OpenGL");
            continue;
        }
        let shown: Vec<_> = rounded
            .iter()
            .take(4)
            .map(|&n| {
                let at = (n as u32 % WIDTH, n as u32 / WIDTH);
                (at, &unwrapped[n * 4..n * 4 + 4], &b[n * 4..n * 4 + 4])
            })
            .collect();
        assert!(
            most <= 1 && rounded.len() <= pixels / 10_000,
            "case {i}: {} pixels differ from the full shaders' without their wrap, by up to \
             {most} (where, full, lean): {shown:?}",
            rounded.len(),
        );
        assert!(
            wrapped.len() <= pixels / 10_000,
            "case {i}: {}",
            wrapped.len()
        );
    }
}

#[test]
fn any_feature_the_lean_shaders_leave_out_keeps_the_full_ones() {
    let mut lean: Globals = bytemuck::Zeroable::zeroed();
    lean.fog = [8.0, 48.0, 30.0, 60.0];
    lean.filtering = [0.0, 16.0, 0.0, 0.0];
    assert!(lean_frame(&lean, true));
    // Anisotropic filtering takes the same one tap, and any fog is the lean shaders' too.
    let mut g = lean;
    g.filtering[0] = 2.0;
    g.fog = [f32::MAX, f32::MAX, 20.0, 10.0];
    assert!(lean_frame(&g, true));
    // A section fading over far terrain, or a distant tile drawn.
    assert!(!lean_frame(&lean, false));
    let features: [fn(&mut Globals); 5] = [
        |g| g.hdr_light[2] = 1.0,
        |g| g.filtering[0] = 1.0,
        |g| g.time[2] = 1.0,
        |g| g.colored_count[0] = 1,
        |g| g.emissive_count[0] = 2,
    ];
    for (i, feature) in features.iter().enumerate() {
        let mut g = lean;
        feature(&mut g);
        assert!(!lean_frame(&g, true), "feature {i}");
    }
}
