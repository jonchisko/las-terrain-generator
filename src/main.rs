use image::ExtendedColorType;
use las::Reader;
use std::error::Error;
use std::num::NonZero;

use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;

const DATA_FILE_PATH: &'static str = "./data/kam_bist.laz";

fn main() -> Result<(), Box<dyn Error>> {
    let mut las_reader = Reader::from_path(DATA_FILE_PATH)?;

    let bounds = las_reader.header().bounds();
    let points_count = las_reader.header().number_of_points();

    let (min_x, min_y, min_z, max_x, max_y, max_z) = (
        bounds.min.x,
        bounds.min.y,
        bounds.min.z,
        bounds.max.x,
        bounds.max.y,
        bounds.max.z,
    );
    let (delta_x, delta_y, delta_z) = (max_x - min_x, max_y - min_y, max_z - min_z);

    println!("Number of points: {}", points_count);
    println!("MIN Bounds: {}, {}, {}", min_x, min_y, min_z);
    println!("MAX Bounds: {}, {}, {}", max_x, max_y, max_z);
    println!("Deltas: {}, {}, {}", delta_x, delta_y, delta_z);

    let (dim_x, dim_y) = (1024usize*4, 1024usize*4);
    let n_channels = 3 * 2;
    let x_side_length = dim_x * n_channels;
    let mut image_buffer: Vec<u8> = vec![1u8; dim_x * dim_y * n_channels];

    let point_data = las_reader
        .points()
        .map(|point| {
            let point = point.unwrap();

            [point.x, point.y, (point.z - min_z) / delta_z]
        })
        .collect::<Vec<[f64; 3]>>();

    let point_data_xy: Vec<[f64; 2]> = point_data
        .iter()
        .map(|point| [point[0], point[1]])
        .collect();

    let kdtree = ImmutableKdTree::<f64, 2>::new_from_slice(&point_data_xy[..]);
    let neighbours_n = 3;
    let nearest_neighbours_n = NonZero::new(neighbours_n).unwrap();

    println!("KDTree size: {}", kdtree.size());

    for linear_index in (0..image_buffer.len()).step_by(n_channels) {
        if linear_index != 0 && linear_index % 50_000 == 0 {
            println!(
                "{} : {}",
                linear_index,
                (linear_index as f32) / (image_buffer.len() as f32)
            );
        }

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

        let height_result = (height_result / neighbours_n as f64 * 65536.0).round() as u16;
        let higher_bits = (height_result >> 8) as u8;
        let lower_bits = height_result as u8;

        image_buffer[index_xy.0 + index_xy.1 * x_side_length] = lower_bits as u8;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 1] = higher_bits as u8;

        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 2] = lower_bits as u8;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 3] = higher_bits as u8;

        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 4] = lower_bits as u8;
        image_buffer[index_xy.0 + index_xy.1 * x_side_length + 5] = higher_bits as u8;
    }

    image::save_buffer(
        "./data/kam_bist.png",
        &image_buffer,
        dim_x as u32,
        dim_y as u32,
        ExtendedColorType::Rgb16,
    )?;

    Ok(())
}
