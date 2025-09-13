use image::ExtendedColorType;
use las::Reader;
use std::error::Error;
use std::num::NonZero;
use std::fs;
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

    println!("Number of data files: {}", file_paths.len());
    println!("Area min height {}, max height {}", min_height, max_height);

    let cpus = thread::available_parallelism()?;
    let work_amount = file_paths.len() / cpus;

    println!("Number of CPUs: {}, average work per thread: {}", cpus, work_amount);

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

fn create_texture(file_path: &PathBuf, min_height: f64, max_height: f64) -> Result<(), Box<dyn Error>> {
    let mut las_reader = Reader::from_path(file_path)?;
    let bounds = las_reader.header().bounds();

    let (min_x, min_y, max_x, max_y) = (
        bounds.min.x,
        bounds.min.y,
        bounds.max.x,
        bounds.max.y,
    );

    let (delta_x, delta_y) = (max_x - min_x, max_y - min_y);

    let (dim_x, dim_y) = (1024usize, 1024usize);
    let n_channels = 3 * 4;
    let x_side_length = dim_x * n_channels;
    let mut image_buffer: Vec<u8> = vec![1u8; dim_x * dim_y * n_channels];

    let point_data = las_reader
        .points()
        .map(|point| {
            let point = point.unwrap();

            [point.x, point.y, (point.z - min_height) / (max_height - min_height)]
        })
        .collect::<Vec<[f64; 3]>>();

    let point_data_xy: Vec<[f64; 2]> = point_data
        .iter()
        .map(|point| [point[0], point[1]])
        .collect();

    let kdtree = ImmutableKdTree::<f64, 2>::new_from_slice(&point_data_xy[..]);
    let neighbours_n = 3;
    let nearest_neighbours_n = NonZero::new(neighbours_n).unwrap();

    for linear_index in (0..image_buffer.len()).step_by(n_channels) {
        /*if linear_index != 0 && linear_index % 50_000 == 0 {
            println!(
                "{} : {}",
                linear_index,
                (linear_index as f32) / (image_buffer.len() as f32)
            );
        }*/

        let index_xy = (linear_index % x_side_length, linear_index / x_side_length);
        let (ratio_x, ratio_y) = (
            (index_xy.0 as f64) / x_side_length as f64,
            (dim_y as f64 - index_xy.1 as f64) / dim_y as f64,
        );

        let (geo_x, geo_y) = (
            ratio_x * delta_x as f64 + min_x,
            ratio_y * delta_y as f64 + min_y,
        );

        let nearest_neighbours = kdtree.nearest_n::<SquaredEuclidean>(&[geo_x, geo_y], nearest_neighbours_n);
        let mut height_result = 0f64;
        
        for neighbour in nearest_neighbours {
            height_result += point_data[neighbour.item as usize][2];
        }

        let height_result = (height_result / neighbours_n as f64 * 4294967296.0).round() as u32;
        let byte_0 = height_result as u8;
        let byte_1 = (height_result >> 8) as u8;
        let byte_2 = (height_result >> 16) as u8;
        let byte_3 = (height_result >> 24) as u8;

        image_buffer[index_xy.0 + index_xy.1 * x_side_length] = byte_0;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 1] = byte_1;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 2] = byte_2;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 3] = byte_3;

        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 4] = byte_0;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 5] = byte_1;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 6] = byte_2;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 7] = byte_3;

        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 8] = byte_0;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 9] = byte_1;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 10] = byte_2;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 11] = byte_3;
    }

    let mut naming = file_path.file_stem().unwrap().to_str().unwrap().split('_').skip(1);
    let x_id = naming.next().unwrap();
    let y_id = naming.next().unwrap();

    image::save_buffer(
        format!("{}{}_{}.exr", RESULTS_DIR_PATH, x_id, y_id),
        &image_buffer,
        dim_x as u32,
        dim_y as u32,
        ExtendedColorType::Rgb32F,
    )?;

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
