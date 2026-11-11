//! Horizon: native Rust client for Minecraft Java Edition 26.3 gameplay.
#![forbid(unsafe_code)]

#[cfg(not(any(feature = "system-allocator", feature = "alloc-census")))]
#[global_allocator]
static ALLOCATOR: horizon_threads::Allocator = horizon_threads::ALLOCATOR;
#[cfg(all(feature = "alloc-census", not(feature = "system-allocator")))]
#[global_allocator]
static ALLOCATOR: horizon_threads::CensusAllocator = horizon_threads::CensusAllocator;

/// The mod API level this game serves, kept in the program for the launcher
/// to read without running it (`horizon_mods::api_level::MARK`). `main`
/// reads it, so the linker keeps it.
static MOD_API_LEVEL: [u8; 27] = horizon_mods::api_level::MARK;

/// SDL games get this machine's pad layout from `SDL_GAMECONTROLLERCONFIG_FILE`. gilrs doesn't read
/// that file, and only `SDL_GAMECONTROLLERCONFIG` wins over the SDL database it bundles, so the file's
/// rows for the pads this machine has go in that variable. Before any thread starts.
fn adopt_sdl_mapping_file() {
    let Some(file) = std::env::var_os("SDL_GAMECONTROLLERCONFIG_FILE") else {
        return;
    };
    let Ok(rows) = std::fs::read_to_string(file) else {
        return;
    };
    let config = std::env::var("SDL_GAMECONTROLLERCONFIG").ok();
    let pads = input_device_guids();
    if let Some(config) =
        horizon_client::gamepad::sdl_config_with_file(&rows, &pads, config.as_deref())
    {
        horizon_threads::set_env_before_threads("SDL_GAMECONTROLLERCONFIG", &config);
    }
}

/// SDL's GUID of each input device the kernel has now.
fn input_device_guids() -> Vec<String> {
    let Ok(devices) = std::fs::read_dir("/sys/class/input") else {
        return Vec::new();
    };
    devices
        .flatten()
        .filter_map(|device| {
            let id = |name: &str| {
                let path = device.path().join("id").join(name);
                u16::from_str_radix(std::fs::read_to_string(path).ok()?.trim(), 16).ok()
            };
            Some(horizon_client::gamepad::sdl_guid([
                id("bustype")?,
                id("vendor")?,
                id("product")?,
                id("version")?,
            ]))
        })
        .collect()
}

fn main() {
    adopt_sdl_mapping_file();
    std::hint::black_box(&MOD_API_LEVEL);
    horizon_client::entry::main();
}
