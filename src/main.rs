use std::error::Error;
use std::thread;

mod computer;
mod core;
mod global_constants;
mod requester;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let config = core::read_config_from_cli()?;

    let cpus = thread::available_parallelism()?;
    let laz_binary_data = requester::get_laz_data(cpus, &config);

    computer::compute_textures_parallel(&config, cpus, laz_binary_data)?;

    Ok(())
}
