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

pub fn get_laz_data(cpus: NonZero<usize>, config: &Config) -> Vec<Reader> {
    let points = filter_points(&config);
    let mut laz_readers: Vec<Reader> = Vec::new();

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

                    let response = client.get(&url).send();

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

                    tx.send(Cursor::new(data_bytes.unwrap()))
                        .expect(&format!("Issue in thread: '{}', in tx send", id));

                    thread::sleep(Duration::from_secs(1 * rand::thread_rng().gen_range(0..5)));
                    // If you find the right block, x, y combination, you got the point. Thus you can move to the next one (break the loop)
                    break;
                }

                access_index += cpus.get();
            }
        });
    }

    drop(tx);

    for received in rx {
        laz_readers.push(Reader::new(received).unwrap());
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
