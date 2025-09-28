use itertools::Itertools;
use las::Reader;
use rand::Rng;
use reqwest::blocking::Client;
use std::num::NonZero;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::{io::Cursor, sync::mpsc};

use crate::core::Config;
use crate::core::Point;
use crate::global_constants::{MAX_POINT_DIM, MIN_POINT_DIM};

pub struct LazData {
    pub offset_from_center: (i16, i16),
    pub bounds_max: (f64, f64, f64),
    pub bounds_min: (f64, f64, f64),
    pub points: Vec<las::Point>,
}

pub fn get_laz_data(cpus: NonZero<usize>, config: &Config) -> Vec<LazData> {
    let points = filter_points(&config);
    let mut laz_readers: Vec<LazData> = Vec::new();

    let coordinate_origin = points.first().expect("There is no points");
    let coordinate_origin = (coordinate_origin.0, coordinate_origin.1);

    let shared_points = Arc::new(points);
    let shared_blocks = Arc::new(
        config
            .possible_blocks
            .iter()
            .map(|e| *e)
            .unique()
            .collect::<Vec<u8>>(),
    );

    let (tx, rx) = mpsc::channel();

    for id in 0..cpus.get() {
        let shared_points = Arc::clone(&shared_points);
        let shared_blocks = Arc::clone(&shared_blocks);
        let tx = tx.clone();

        thread::spawn(move || {
            let mut access_index = id;
            let client = Client::new();

            loop {
                if access_index >= shared_points.len() {
                    break;
                }

                let point = &shared_points[access_index];

                for block_number in shared_blocks.iter() {
                    println!("Point {}:{}|block {}", point.0, point.1, block_number);

                    let url = format!(
                        "https://gis.arso.gov.si/lidar/otr/laz/b_{}/D96TM/TMR_{}_{}.laz",
                        block_number, point.0, point.1
                    );

                    let response = client.get(&url).timeout(Duration::from_secs(300)).send();

                    if response.is_err() {
                        println!("HTTP get not successful, error. Skipping point url {}", url);
                        continue;
                    }

                    let response = response.unwrap();

                    if !response.status().is_success() {
                        println!(
                            "HTTP status not successful (not 200 OK). Skipping point url {}",
                            url
                        );
                        continue;
                    }

                    let data_bytes = response.bytes();

                    if let Err(value) = data_bytes {
                        println!("Err: {}", value);
                        println!(
                            "Reading bytes was not successful. Skipping point url {}",
                            url
                        );
                        continue;
                    }

                    let offset_from_center =
                        (point.0 - coordinate_origin.0, point.1 - coordinate_origin.1);

                    let mut laz_reader = Reader::new(Cursor::new(data_bytes.unwrap())).unwrap();
                    let bounds = laz_reader.header().bounds();
                    let points = laz_reader.points().collect::<Result<Vec<_>, _>>().unwrap();

                    tx.send((offset_from_center, bounds, points))
                        .expect(&format!("Issue in thread: '{}', in tx send", id));

                    thread::sleep(Duration::from_secs(1 * rand::thread_rng().gen_range(0..5)));
                    // If you find the right block, x, y combination, you got the point. Thus you can move to the next one (break the loop)
                    break;
                }

                access_index += cpus.get();
            }
        });
    }

    // Last TX must be dropped to ensure rx does not continue listening
    drop(tx);

    for received in rx {
        laz_readers.push(LazData {
            offset_from_center: received.0,
            bounds_max: (received.1.max.x, received.1.max.y, received.1.max.z),
            bounds_min: (received.1.min.x, received.1.min.y, received.1.min.z),
            points: received.2,
        });
    }

    laz_readers
}

fn filter_points(config: &Config) -> Vec<Point> {
    config
        .core_points
        .iter()
        .flat_map(|core_point| core_point.get_all_points_in_area())
        .filter(|point| {
            point.0 >= MIN_POINT_DIM
                && point.1 >= MIN_POINT_DIM
                && point.0 < MAX_POINT_DIM
                && point.1 < MAX_POINT_DIM
        })
        .unique()
        .collect()
}
