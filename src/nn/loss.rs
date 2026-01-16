use wasm_bindgen::prelude::*;

use crate::core::tensor::Tensor;

const EPS: f32 = 1e-7;

#[wasm_bindgen]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub enum LOSS {
    DEFAULT,
    BINARY_CROSS_ENTROPY,
    QUAD
}

impl Default for LOSS {
    fn default() -> Self {
        Self::DEFAULT
    }
}

pub fn get_function(loss: LOSS) -> fn(&Tensor, &Tensor) -> f32 {
    use LOSS::*;
    match loss {
        DEFAULT => default,
        BINARY_CROSS_ENTROPY => binary_cross_entropy,
        QUAD => quad
    }
}

pub fn get_prime(activation: LOSS) -> fn(&Tensor, &Tensor) -> Tensor {
    use LOSS::*;
    match activation {
        DEFAULT => default_prime,
        BINARY_CROSS_ENTROPY => binary_cross_entropy_prime,
        QUAD => quad_prime
    }
}

fn default(prediction: &Tensor, target: &Tensor) -> f32 {
    let target_slice = target.as_slice().unwrap();
    let mut count = 0;
    let loss: f32 = prediction.iter().map(|x| {
        let result = target_slice[count] - x;
        count += 1;
        result.abs()
    }).sum();
    loss / target.len() as f32
}

fn default_prime(prediction: &Tensor, target: &Tensor) -> Tensor {
    let target_slice = target.as_slice().unwrap();
    let mut count = 0;
    prediction.map(|x| {
        let sign = (x - target_slice[count]).signum() as f32;
        count += 1;
        sign / prediction.len() as f32
    }) 
}

fn quad(prediction: &Tensor, target: &Tensor) -> f32 {
    let target_slice = target.as_slice().unwrap();
    let mut count = 0;
    let loss: f32 = prediction.iter().map(|x| {
        let result = target_slice[count] - x;
        count += 1;
        result.powi(2)
    }).sum();
    loss / target.len() as f32
}

fn quad_prime(prediction: &Tensor, target: &Tensor) -> Tensor {
    let target_slice = target.as_slice().unwrap();
    let mut count = 0;
    prediction.map(|x| {
        let result = x - target_slice[count];
        count += 1;
        2.0 * result / prediction.len() as f32
    }) 
}

fn binary_cross_entropy(prediction: &Tensor, target: &Tensor) -> f32 {
    let prediction = prediction.as_f32();
    let target = target.as_f32();
    let prediction = prediction.clamp(EPS, 1.0 - EPS);
    let result = target * prediction.ln() + (1.0 - target) *  (1.0 - prediction).ln();
    -result
}

fn binary_cross_entropy_prime (prediction: &Tensor, target: &Tensor) -> Tensor {
    prediction - target
}