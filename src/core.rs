use std::{error::Error, fmt::Display, str::FromStr};

use clap::Parser;

#[derive(Clone, Copy, Debug)]
pub enum CommandlineParsingErrors {
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

pub struct CorePointIterator {
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
pub struct CorePoint {
    center: Point,
    radius: u8,
}

impl CorePoint {
    pub fn new(center: Point, radius: u8) -> Self {
        CorePoint { center, radius }
    }

    pub fn get_all_points_in_area(&self) -> CorePointIterator {
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Point(pub i16, pub i16);

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

pub struct Config {
    pub core_points: Vec<CorePoint>,
    pub possible_blocks: Vec<u8>,
    pub blur_kernel_size: u8,
    pub sample_size: u8,
    pub resolution: u16,
    pub destination_folder: String,
}

impl Config {
    fn new(
        core_points: Vec<CorePoint>,
        possible_blocks: Vec<u8>,
        blur_kernel_size: u8,
        sample_size: u8,
        resolution: u16,
        destination_folder: String,
    ) -> Self {
        Config {
            core_points,
            possible_blocks,
            blur_kernel_size,
            sample_size,
            resolution,
            destination_folder,
        }
    }
}

impl TryFrom<&Cli> for Config {
    type Error = CommandlineParsingErrors;

    fn try_from(value: &Cli) -> Result<Self, Self::Error> {
        Ok(Config::new(
            Vec::<CorePoint>::try_from(value)?,
            value.possible_blocks.clone(),
            value.blur_kernel_size,
            value.sample_size,
            value.resolution,
            value.destination_folder.clone(),
        ))
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
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

pub fn read_config_from_cli() -> Result<Config, CommandlineParsingErrors> {
    let arguments = Cli::parse();

    if arguments.points.len() != arguments.radius.len() {
        return Err(CommandlineParsingErrors::NumberOfPointsAndRadius(
            "Number of points must equal number of radius-es",
        ));
    }

    if arguments.possible_blocks.len() < 1 {
        return Err(CommandlineParsingErrors::IncorrectArgumentStructure(
            "At least one possible block must be given",
        ));
    }

    Config::try_from(&arguments)
}
