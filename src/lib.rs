mod utils;

pub mod core;
pub mod nn;
pub mod data;
pub mod wasm;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, neural_network!");
}
