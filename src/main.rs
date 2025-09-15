use exr::image::Encoding;
use exr::image::SpecificChannels;
use exr::math::Vec2;
use exr::prelude::LayerAttributes;
use exr::prelude::WritableImage;
use las::Reader;
use libblur::AnisotropicRadius;
use libblur::BlurImageMut;
use libblur::EdgeMode;
use libblur::EdgeMode2D;
use libblur::ThreadingPolicy;
use std::error::Error;
use std::fs;
use std::num::NonZero;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;

const DATA_DIR_PATH: &'static str = "./data/";
const RESULTS_DIR_PATH: &'static str = "./results/";

fn main() -> Result<(), Box<dyn Error>> {
    let file_paths = get_files_in_directory()?;
    let (min_height, max_height) = get_height_bounds(&file_paths)?;
    let cpus = thread::available_parallelism()?;
    let work_amount = file_paths.len() / cpus;

    println!("Number of data files: {}", file_paths.len());
    println!("Area min height {}, max height {}", min_height, max_height);
    println!(
        "Number of CPUs: {}, average work per thread: {}",
        cpus, work_amount
    );

    let mut handles = vec![];
    let shared_file_paths = Arc::new(file_paths);

    for id in 0..cpus.get() {
        let shared_file_paths = Arc::clone(&shared_file_paths);

        handles.push(thread::spawn(move || {
            let mut access_index = id;
            loop {
                if access_index >= shared_file_paths.len() {
                    break;
                }

                let file_path = &shared_file_paths[access_index];
                create_texture(&file_path, min_height, max_height).unwrap();

                access_index += cpus.get();
            }
        }));
    }

    for handle in handles {
        handle.join().map_err(|_| format!("Thread error"))?;
    }

    Ok(())
}

fn create_texture(
    file_path: &PathBuf,
    min_height: f64,
    max_height: f64,
) -> Result<(), Box<dyn Error>> {
    let mut las_reader = Reader::from_path(file_path)?;
    let bounds = las_reader.header().bounds();

    let (min_x, min_y, max_x, max_y) = (bounds.min.x, bounds.min.y, bounds.max.x, bounds.max.y);

    let (delta_x, delta_y) = (max_x - min_x, max_y - min_y);

    let dim_multiplier = 2;
    let channel_num = 3;
    let (dim_x, dim_y) = (1024usize * dim_multiplier, 1024usize * dim_multiplier);
    let dim_x_adapted = dim_x * channel_num;

    let point_data = las_reader
        .points()
        .map(|point| {
            let point = point.unwrap();

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
    let neighbours_n = 3;
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

    let mut blured_image = BlurImageMut::borrow(
        &mut buffer_f32,
        dim_x as u32,
        dim_y as u32,
        libblur::FastBlurChannels::Channels3,
    );

    libblur::fast_gaussian_f32(
        &mut blured_image,
        AnisotropicRadius {
            x_axis: 10,
            y_axis: 10,
        },
        ThreadingPolicy::Adaptive,
        EdgeMode2D::anisotropy(EdgeMode::Wrap, EdgeMode::Wrap),
    )?;

    let channels = SpecificChannels::rgb(|position: Vec2<usize>| {
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

    let file_name = create_indexed_name(file_path);

    image.write().to_file(file_name)?;

    Ok(())
}

fn get_height_bounds(file_paths: &Vec<std::path::PathBuf>) -> Result<(f64, f64), Box<dyn Error>> {
    let (mut min_height, mut max_height) = (f64::MAX, f64::MIN);

    for path in file_paths {
        let las_reader = Reader::from_path(path)?;
        let bounds = las_reader.header().bounds();

        if min_height > bounds.min.z {
            min_height = bounds.min.z;
        }

        if max_height < bounds.max.z {
            max_height = bounds.max.z;
        }
    }

    Ok((min_height, max_height))
}

fn get_files_in_directory() -> Result<Vec<std::path::PathBuf>, Box<dyn Error>> {
    let mut file_paths = vec![];

    for path in fs::read_dir(DATA_DIR_PATH)? {
        if let Ok(path) = path {
            if !path.metadata().is_ok_and(|metadata| metadata.is_file()) {
                continue;
            }

            file_paths.push(path.path());
        }
    }

    Ok(file_paths)
}

fn create_indexed_name(file_path: &PathBuf) -> String {
    let mut naming = file_path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .split('_')
        .skip(1);

    let x_id = naming.next().unwrap();
    let y_id = naming.next().unwrap();

    format!("{}{}_{}.exr", RESULTS_DIR_PATH, x_id, y_id)
}
