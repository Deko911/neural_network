use core::{f32, f64};
use std::fs::File;

use serde_json::Value;


pub fn read_json(file_path: &str) -> Value {
    let file = File::open(file_path).expect("File doesn't found");
    let json = serde_json::from_reader(&file).expect("JSON was not well-formatted");
    json  
}

pub fn read_csv(file_path: &str) -> Vec<Vec<String>> {
    let file = File::open(file_path).expect("File doesn't found");
    let mut rdr = csv::Reader::from_reader(file);
    rdr.records().map(|r| r.expect("Invalid record").iter().map(|x| String::from(x)).collect()).collect()
}

pub fn json_to_array(data: &Value) -> Option<Vec<f32>> {
    let data = data.as_array();
    if let None = data {
        return None;
    }
    let result: Vec<f32> = data.unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN) as f32).collect();
    Some(result)
}

pub fn json_to_matrix(data: &Value) -> Option<Vec<Vec<f32>>> {
    let data = data.as_array();
    if let None = data {
        return None;
    }
    let iter = data.unwrap().iter();
    let mut result: Vec<Vec<f32>> = vec![];
    for i in iter {
        let arr = i.as_array();
        if let None = arr {
            return None;
        }
        result.push(json_to_array(i).unwrap());
    }
    Some(result)
}

pub fn get_column(data: &Vec<Vec<String>>, idx: usize) -> Vec<f32> {
    data.iter().map(|r| r[idx].trim().parse::<f32>().unwrap_or(f32::NAN)).collect()
}

pub fn csv_to_matrix(data: &Vec<Vec<String>>, size: usize) -> Vec<Vec<f32>> {
    let mut result: Vec<Vec<f32>> = vec![];
    for r in 0..data.len() {
        result.push(vec![]);
        for c in 0..size {
            result[r].push(data[r][c].trim().parse::<f32>().unwrap_or(f32::NAN));
        }
    }
    result
}

