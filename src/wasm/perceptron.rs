use crate::core::tensor::Tensor;
use crate::nn::activation::ACTIVATIONS;
use crate::nn::model::Model;
use crate::nn::perceptron::PerceptronModel;
use crate::utils::set_panic_hook;
use crate::wasm::utils::parse_array_to_tensor;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct PerceptronJS {
    inner: PerceptronModel,
    input_size: usize,
}

#[wasm_bindgen]
impl PerceptronJS {
    #[wasm_bindgen(constructor)]
    pub fn new(input_size: usize, lr: f32, activation: Option<ACTIVATIONS>) -> Self {
        set_panic_hook();
        let inner = PerceptronModel::new(input_size, lr, activation);
        Self { inner, input_size }
    }

    #[wasm_bindgen]
    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        let input = Tensor::from_vec(input);
        self.inner.predict(&input).to_vec()
    }

    #[wasm_bindgen]
    pub fn evaluate(&self, input: Vec<f32>, target: &[f32]) -> f32 {
        let input = parse_array_to_tensor(self.input_size, input);
        let target = &Tensor::from_vec_col(target.to_vec());
        self.inner.evaluate(&input, target)
    }

    #[wasm_bindgen]
    pub fn evaluate_one(&self, input: Vec<f32>, target: f32) -> f32 {
        let input = Tensor::from_vec(input);
        let target = &Tensor::from_elem(target);
        self.inner.evaluate_one(&input, &target)
    }

    #[wasm_bindgen]
    pub fn fit(&mut self, input: Vec<f32>, target: &[f32], epochs: usize) {
        let input = parse_array_to_tensor(self.input_size, input);
        let target = &Tensor::from_vec_col(target.to_vec());
        self.inner.fit(&input, target, epochs);
    }

    #[wasm_bindgen]
    pub fn fit_raw(&mut self, input: Vec<f32>, target: &[f32], epochs: usize) {
        let input = parse_array_to_tensor(self.input_size, input);
        let target = &Tensor::from_vec_col(target.to_vec());
        self.inner.fit_raw(&input, target, epochs);
    }
}
