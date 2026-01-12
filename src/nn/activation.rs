use wasm_bindgen::prelude::*;

use crate::core::tensor::Tensor;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ACTIVATIONS {
    DEFAULT,
    SIGMOID,
}

impl Default for ACTIVATIONS {
    fn default() -> Self {
        ACTIVATIONS::DEFAULT
    }
}

pub fn get_function(activation: ACTIVATIONS) -> fn(Tensor) -> Tensor {
    use ACTIVATIONS::*;
    match activation {
        DEFAULT => default,
        SIGMOID => sigmoid
    }
}

pub fn default(x: Tensor) -> Tensor {
    x
}

pub fn sigmoid(x: Tensor) -> Tensor {
    x.map(|el| 1.0 / (1.0 + (-el).exp()))
}
