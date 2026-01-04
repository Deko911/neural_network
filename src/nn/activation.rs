use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub enum ACTIVATIONS {
    DEFAULT,
    SIGMOID,
}

impl Default for ACTIVATIONS {
    fn default() -> Self {
        ACTIVATIONS::DEFAULT
    }
}

pub fn get_function(activation: ACTIVATIONS) -> fn(f32) -> f32 {
    use ACTIVATIONS::*;
    match activation {
        DEFAULT => default,
        SIGMOID => sigmoid
    }
}

pub fn default(x: f32) -> f32 {
    x
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
