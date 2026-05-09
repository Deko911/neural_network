use wasm_bindgen::prelude::*;

use crate::core::tensor::Tensor;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ACTIVATIONS {
    LINEAR,
    SIGMOID,
    RELU,
    SOFTMAX
}

impl Default for ACTIVATIONS {
    fn default() -> Self {
        ACTIVATIONS::LINEAR
    }
}

impl ACTIVATIONS {
    pub fn get_function(activation: ACTIVATIONS) -> fn(&Tensor) -> Tensor {
        use ACTIVATIONS::*;
        match activation {
            LINEAR => linear,
            SIGMOID => sigmoid,
            RELU => relu,
            SOFTMAX => softmax
        }
    }

    pub fn get_prime(activation: ACTIVATIONS) -> fn(&Tensor) -> Tensor {
        use ACTIVATIONS::*;
        match activation {
            LINEAR => linear_prime,
            SIGMOID => sigmoid_prime,
            RELU => relu_prime,
            SOFTMAX => linear_prime // This cannot happen
        }
    }
}

fn linear(x: &Tensor) -> Tensor {
    x.clone()
}

fn sigmoid(x: &Tensor) -> Tensor {
    x.map(|el| 1.0 / (1.0 + (-el).exp()))
}

fn relu(x: &Tensor) -> Tensor {
    x.map(|el| el.max(0.0))
}

fn softmax(x: &Tensor) -> Tensor {
    use std::f32::consts::E;
    let mut max_x = f32::NEG_INFINITY;
    x.iter().for_each(|el| {
        if *el > max_x{
            max_x = *el
        }
    });
    let sum: f32 = x.iter().map(|el| E.powf(el - max_x)).sum();
    x.map(|el| E.powf(el - max_x) / sum)
}

fn linear_prime(_: &Tensor) -> Tensor {
    Tensor::from_elem(1.0)
}

fn sigmoid_prime(z: &Tensor) -> Tensor {
    let sig = sigmoid(z);
    sig.map(|el| el * (1.0 - el))
}

fn relu_prime(z: &Tensor) -> Tensor {
    z.map(|el| if *el > 0.0 {1.0} else {0.0})
}