use wasm_bindgen::prelude::*;

use crate::core::tensor::Tensor;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ACTIVATIONS {
    LINEAR,
    SIGMOID,
}

impl Default for ACTIVATIONS {
    fn default() -> Self {
        ACTIVATIONS::LINEAR
    }
}

pub fn get_function(activation: ACTIVATIONS) -> fn(&Tensor) -> Tensor {
    use ACTIVATIONS::*;
    match activation {
        LINEAR => linear,
        SIGMOID => sigmoid
    }
}

pub fn get_prime(activation: ACTIVATIONS) -> fn(&Tensor) -> Tensor {
    use ACTIVATIONS::*;
    match activation {
        LINEAR => linear_prime,
        SIGMOID => sigmoid_prime
    }
}

fn linear(x: &Tensor) -> Tensor {
    x.clone()
}

fn sigmoid(x: &Tensor) -> Tensor {
    x.map(|el| 1.0 / (1.0 + (-el).exp()))
}

fn linear_prime (_: &Tensor) -> Tensor {
    Tensor::from_elem(1.0)
}

fn sigmoid_prime(z: &Tensor) -> Tensor {
    let sig = sigmoid(z);
    sig.map(|el| el * (1.0 - el))
}