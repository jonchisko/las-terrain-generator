use std::{error::Error, fs, num::NonZero, thread};

use exr::{
    image::{Encoding, Image, Layer, SpecificChannels},
    math::Vec2,
    prelude::{ChannelDescription, LayerAttributes, WritableImage},
};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use libblur::{AnisotropicRadius, BlurImageMut, EdgeMode, EdgeMode2D, ThreadingPolicy};
use serde::Serialize;

use crate::{core::Config, requester::LazData};

#[derive(Serialize)]
struct ComputeConfig {
    texture_resolution: u16,
    max_height: f64,
    min_height: f64,
    real_world_dimensions_m: f64,
}

pub fn compute_textures_parallel(
    config: &Config,
    cpus: NonZero<usize>,
    data: Vec<LazData>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (min_height, max_height) = get_height_bounds(&data)?;
    let work_amount = data.len() / cpus + 1;

    println!("Number of data elements: {}", data.len());
    println!("Area min height {}, max height {}", min_height, max_height);
    println!(
        "Number of CPUs: {}, average work per thread: {}",
        cpus, work_amount
    );

    thread::scope(|scope| -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut results = vec![];
        for (_id, chunk) in data.chunks(work_amount).enumerate() {
            let result = scope.spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
                for data in chunk {
                    create_texture(config, data, min_height, max_height)?;
                }

                Ok(())
            });

            results.push(result);
        }

        for result in results {
            result.join().unwrap();
        }

        Ok(())
    })?;

    let cfg = ComputeConfig {
        texture_resolution: config.resolution,
        max_height: max_height,
        min_height: min_height,
        real_world_dimensions_m: 1000.0,
    };

    let json = serde_json::to_string_pretty(&cfg)?;

    println!("Writing meta data.");
    fs::write(format!("{}/config.json", config.destination_folder), json)?;

    Ok(())
}

fn create_texture(
    config: &Config,
    data: &LazData,
    min_height: f64,
    max_height: f64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (min_x, min_y, max_x, max_y) = (
        data.bounds_min.0,
        data.bounds_min.1,
        data.bounds_max.0,
        data.bounds_max.1,
    );
    let (delta_x, delta_y) = (max_x - min_x, max_y - min_y);

    let channel_num = 3;
    let (dim_x, dim_y) = (config.resolution as usize, config.resolution as usize);
    let dim_x_adapted = dim_x * channel_num;

    let point_data = data
        .points
        .iter()
        .map(|point| {
            let point = point;

            [
                point.x,
                point.y,
                (point.z - min_height) / (max_height - min_height),
            ]
        })
        .collect::<Vec<[f64; 3]>>();

    let point_data_xy: Vec<[f64; 2]> = point_data
        .iter()
        .map(|point| [point[0], point[1]])
        .collect();

    let kdtree = ImmutableKdTree::<f64, 2>::new_from_slice(&point_data_xy[..]);
    let neighbours_n = config.sample_size as usize;
    let nearest_neighbours_n = NonZero::new(neighbours_n).unwrap();

    let mut buffer_f32: Vec<f32> = vec![0f32; dim_x_adapted * dim_y];

    for linear_index in (0..(dim_x_adapted * dim_y)).step_by(channel_num) {
        let (ind_x, ind_y) = (linear_index % dim_x_adapted, linear_index / dim_x_adapted);
        let (ind_x, ind_y) = (ind_x, dim_y - ind_y);

        let (geo_x, geo_y) = (
            (ind_x as f64 / dim_x_adapted as f64) * delta_x as f64 + min_x,
            (ind_y as f64 / dim_y as f64) * delta_y as f64 + min_y,
        );

        let nearest_neighbours =
            kdtree.nearest_n::<SquaredEuclidean>(&[geo_x, geo_y], nearest_neighbours_n);
        let mut height_result = 0f32;

        for neighbour in nearest_neighbours {
            height_result += point_data[neighbour.item as usize][2] as f32;
        }

        let height_result = height_result / neighbours_n as f32;

        buffer_f32[linear_index] = height_result;
        buffer_f32[linear_index + 1] = height_result;
        buffer_f32[linear_index + 2] = height_result;
    }

    blur_image(
        config.blur_kernel_size as u32,
        dim_x,
        dim_y,
        &mut buffer_f32,
    )?;

    let image = create_image(channel_num, dim_x, dim_y, dim_x_adapted, &buffer_f32);

    let file_coord_name_x = get_coordinate_name(data.offset_from_center.0);
    let file_coord_name_y = get_coordinate_name(data.offset_from_center.1);
    let file_path = format!(
        "{}/img_{}_{}.exr",
        config.destination_folder, file_coord_name_x, file_coord_name_y
    );

    image.write().to_file(file_path)?;

    Ok(())
}

fn get_coordinate_name(value: i16) -> String {
    if value < 0 {
        "n".to_string() + &value.abs().to_string()
    } else {
        value.to_string()
    }
}

fn create_image<'a>(
    channel_num: usize,
    dim_x: usize,
    dim_y: usize,
    dim_x_adapted: usize,
    buffer_f32: &'a Vec<f32>,
) -> Image<
    Layer<
        SpecificChannels<
            impl Fn(Vec2<usize>) -> (f32, f32, f32),
            (ChannelDescription, ChannelDescription, ChannelDescription),
        >,
    >,
> {
    let channels = SpecificChannels::rgb(move |position: Vec2<usize>| {
        let linear_index = position.0 * channel_num + position.1 * dim_x_adapted;
        let data = buffer_f32[linear_index];

        (data, data, data)
    });

    let image = exr::prelude::Image::from_layer(exr::prelude::Layer::new(
        (dim_x, dim_y),
        LayerAttributes::named("main-rgb-layer"),
        Encoding::SMALL_LOSSLESS,
        channels,
    ));

    image
}

fn blur_image(
    kernel_size: u32,
    dim_x: usize,
    dim_y: usize,
    buffer_f32: &mut Vec<f32>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut blured_image = BlurImageMut::borrow(
        buffer_f32,
        dim_x as u32,
        dim_y as u32,
        libblur::FastBlurChannels::Channels3,
    );

    libblur::fast_gaussian_f32(
        &mut blured_image,
        AnisotropicRadius {
            x_axis: kernel_size,
            y_axis: kernel_size,
        },
        ThreadingPolicy::Adaptive,
        EdgeMode2D::anisotropy(EdgeMode::Clamp, EdgeMode::Clamp),
    )?;

    Ok(())
}

fn get_height_bounds(data: &[LazData]) -> Result<(f64, f64), Box<dyn Error + Send + Sync>> {
    let (mut min_height, mut max_height) = (f64::MAX, f64::MIN);

    for sector in data {
        if min_height > sector.bounds_min.2 {
            min_height = sector.bounds_min.2;
        }

        if max_height < sector.bounds_max.2 {
            max_height = sector.bounds_max.2
        }
    }

    Ok((min_height, max_height))
}
