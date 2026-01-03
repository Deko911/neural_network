use crate::{nn::{model::Model, perceptron::PerceptronModel}, utils::set_panic_hook};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct PerceptronJS  {
    inner: PerceptronModel,
    input_size: usize
}

impl PerceptronJS {
    fn parse_array_to_matrix(&self, input: Vec<f32>) -> Vec<Vec<f32>>{
        let mut input_parsed: Vec<Vec<f32>> = vec![];
        for i in 0..input.len() {
            if i % self.input_size == 0 {
                input_parsed.push(vec![input[i]]);
                continue;
            }
            input_parsed[i / self.input_size].push(input[i]);
        }
        input_parsed
    }
}

#[wasm_bindgen]
impl PerceptronJS {
    #[wasm_bindgen(constructor)]
    pub fn new(input_size: usize, lr: f32) -> PerceptronJS {
        set_panic_hook();
        let inner = PerceptronModel::new(input_size, lr);
        Self { inner, input_size }
    }

    #[wasm_bindgen]
    pub fn predict(&self, input: Vec<f32>) -> f32{
        self.inner.predict(&input)
    }

    #[wasm_bindgen]
    pub fn evaluate(&self, input: Vec<f32>, target: &[f32]) -> f32{
        let input = self.parse_array_to_matrix(input);
        self.inner.evaluate(&input, target)
    }

    #[wasm_bindgen]
    pub fn evaluate_one(&self, input: Vec<f32>, target: f32) -> f32{
        self.inner.evaluate_one(&input, &target)
    }

    #[wasm_bindgen]
    pub fn fit(&mut self, input: Vec<f32>, target: &[f32], epochs: usize){
        let input = self.parse_array_to_matrix(input);
        self.inner.fit(&input, target, epochs);
    }

    #[wasm_bindgen]
    pub fn fit_raw(&mut self, input: Vec<f32>, target: &[f32], epochs: usize){
        let input = self.parse_array_to_matrix(input);
        self.inner.fit_raw(&input, target, epochs);
    }
}
