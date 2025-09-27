use clap::Parser;
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
use std::fmt::Display;
use std::fs;
use std::num::NonZero;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::thread;

use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;

const DATA_DIR_PATH: &'static str = "./data/";
const RESULTS_DIR_PATH: &'static str = "./results/";

const MIN_POINT_DIM: u16 = 0;
const MAX_POINT_DIM: u16 = 800;

#[derive(Clone, Copy, Debug)]
enum CommandlineParsingErrors {
    NumberOfPointsAndRadius(&'static str),
    IncorrectArgumentStructure(&'static str),
    IntegerParsing(&'static str),
}

impl Display for CommandlineParsingErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match self {
            CommandlineParsingErrors::NumberOfPointsAndRadius(msg) => {
                format!("Number of points and radius, {}", msg)
            }
            CommandlineParsingErrors::IncorrectArgumentStructure(msg) => {
                format!("Incorrect argument structure, {}", msg)
            }
            CommandlineParsingErrors::IntegerParsing(msg) => format!("Integer parsing, {}", msg),
        };

        f.pad(&result)
    }
}

impl Error for CommandlineParsingErrors {}

struct CorePointIterator {
    start_position: (i16, i16),
    current_position_index: (u8, u8),
    side_dimension: u8,
}

impl Iterator for CorePointIterator {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_position_index.0 >= self.side_dimension
            || self.current_position_index.1 >= self.side_dimension
        {
            return None;
        }

        let current_point = Point(
            self.start_position.0 + self.current_position_index.0 as i16,
            self.start_position.1 + self.current_position_index.1 as i16,
        );

        self.current_position_index.0 += 1;
        if self.current_position_index.0 % self.side_dimension == 0 {
            self.current_position_index.0 = 0;
            self.current_position_index.1 += 1;
        }

        Some(current_point)
    }
}

#[derive(Clone, Copy)]
struct CorePoint {
    center: Point,
    radius: u8,
}

impl CorePoint {
    fn new(center: Point, radius: u8) -> Self {
        CorePoint { center, radius }
    }

    fn get_all_points_in_area(&self) -> CorePointIterator {
        let side_dimension: u8 = (self.radius as u8)
            .checked_mul(2)
            .and_then(|prod| prod.checked_add(1))
            .or(Some(255))
            .unwrap();

        let (start_x, start_y) = (
            self.center.0 as i16 - self.radius as i16,
            self.center.1 as i16 - self.radius as i16,
        );

        CorePointIterator {
            start_position: (start_x, start_y),
            current_position_index: (0u8, 0u8),
            side_dimension,
        }
    }
}

impl TryFrom<&Cli> for Vec<CorePoint> {
    type Error = CommandlineParsingErrors;

    fn try_from(value: &Cli) -> Result<Self, Self::Error> {
        let mut result = Vec::with_capacity(value.points.len());

        for i in 0..value.points.len() {
            let point = Point::from_str(&value.points[i])?;
            result.push(CorePoint::new(point, value.radius[i]));
        }

        Ok(result)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Point(i16, i16);

impl FromStr for Point {
    type Err = CommandlineParsingErrors;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut coordinates = s.split(",");

        let first_number = coordinates
            .next()
            .and_then(|coord| Some(coord.trim_matches(|ch| ch == '(' || ch == ' ' || ch == ')')));
        let second_number = coordinates
            .next()
            .and_then(|coord| Some(coord.trim_matches(|ch| ch == '(' || ch == ' ' || ch == ')')));

        if None == first_number {
            return Err(CommandlineParsingErrors::IncorrectArgumentStructure(
                "First number is not correctly structured. Structure should be '(x, y)'",
            ));
        }

        if None == second_number {
            return Err(CommandlineParsingErrors::IncorrectArgumentStructure(
                "Second number is not correctly structured. Structure should be '(x, y)'",
            ));
        }

        let first_number = first_number.unwrap().parse::<i16>().map_err(|_err| {
            CommandlineParsingErrors::IntegerParsing("First argument could not be converted to u16")
        })?;
        let second_number = second_number.unwrap().parse::<i16>().map_err(|_err| {
            CommandlineParsingErrors::IntegerParsing("First argument could not be converted to u16")
        })?;

        Ok(Point(first_number, second_number))
    }
}

struct Config {
    possible_blocks: Vec<u8>,
    blur_kernel_size: u8,
    sample_size: u8,
    resolution: u16,
    destination_folder: String,
}

impl Config {
    fn new(
        possible_blocks: Vec<u8>,
        blur_kernel_size: u8,
        sample_size: u8,
        resolution: u16,
        destination_folder: String,
    ) -> Self {
        Config {
            possible_blocks,
            blur_kernel_size,
            sample_size,
            resolution,
            destination_folder,
        }
    }
}

impl From<&Cli> for Config {
    fn from(value: &Cli) -> Self {
        Config::new(
            value.possible_blocks.clone(),
            value.blur_kernel_size,
            value.sample_size,
            value.resolution,
            value.destination_folder.clone(),
        )
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short = 'p', required = true, value_delimiter = ' ', num_args = 1..)]
    points: Vec<String>,

    #[arg(short = 'r', required = true, value_delimiter = ' ', num_args = 1..)]
    radius: Vec<u8>,

    #[arg(long, required = true, value_delimiter = ' ', num_args = 1..)]
    possible_blocks: Vec<u8>,

    #[arg(short = 'b', default_value = "10")]
    blur_kernel_size: u8,

    #[arg(short = 's', default_value = "3")]
    sample_size: u8,

    #[arg(long, default_value = "1024")]
    resolution: u16,

    #[arg(short = 'd', required = true)]
    destination_folder: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Cli::parse();

    if arguments.points.len() != arguments.radius.len() {
        return Err(Box::new(CommandlineParsingErrors::NumberOfPointsAndRadius(
            "Number of points must equal number of radius-es",
        )));
    }

    let core_points = Vec::<CorePoint>::try_from(&arguments).unwrap();
    let config = Config::from(&arguments);

    /*
    Now you have points and the config.
    For each core_point get all points and insert it into a set structure (usually for small numbers a vec is faster),
        if a point is valid - between 0,0 and 600,600
    Now we have all the necessary points

    Compute the number of threads that can be started and divide the load between them.
    Each thread, goes over its points and requests the data - if data not there, continue (some points might not be valid url addresses)
    Then you get the "bytes", basically the LAZ file and store it in your own vector.
    At the end you join the threads and combine all the vectors into one data structure. These vectors reference the bytes or laz files.
    These files are basically bytes hidden behind the cursor, so that reader can read them later on.

    When you have got all this data, you basically do the rest of the already implemented program.
     */

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
