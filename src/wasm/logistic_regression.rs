use crate::core::tensor::Tensor;
use crate::nn::model::Model;
use crate::nn::logistic_regression::LogisticRegression;
use crate::utils::set_panic_hook;
use crate::wasm::utils::parse_array_to_tensor;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LogisticRegressionJS  {
    inner: LogisticRegression,
    input_size: usize
}

#[wasm_bindgen]
impl LogisticRegressionJS {
    #[wasm_bindgen(constructor)]
    pub fn new(input_size: usize, lr: f32, threshold: Option<f32>) -> Self {
        set_panic_hook();
        let inner = LogisticRegression::new(input_size, lr, threshold);
        Self { inner, input_size }
    }

    #[wasm_bindgen]
    pub fn predict(&self, input: Vec<f32>) -> Vec<f32>{
        let input = Tensor::from_vec(input);
        self.inner.predict(&input).to_vec()
    }

    #[wasm_bindgen]
    pub fn evaluate(&self, input: Vec<f32>, target: &[f32]) -> f32{
        let input = parse_array_to_tensor(self.input_size, input);
        let target = &Tensor::from_vec_col(target.to_vec());
        self.inner.evaluate(&input, target)
    }

    #[wasm_bindgen]
    pub fn evaluate_one(&self, input: Vec<f32>, target: f32) -> f32{
        let input = Tensor::from_vec(input);
        let target = &Tensor::from_elem(target);
        self.inner.evaluate_one(&input, &target)
    }

    #[wasm_bindgen]
    pub fn fit(&mut self, input: Vec<f32>, target: &[f32], epochs: usize){
        let input = parse_array_to_tensor(self.input_size, input);
        let target = &Tensor::from_vec_col(target.to_vec());
        self.inner.fit(&input, target, epochs);
    }

    #[wasm_bindgen]
    pub fn fit_raw(&mut self, input: Vec<f32>, target: &[f32], epochs: usize){
        let input = parse_array_to_tensor(self.input_size, input);
        let target = &Tensor::from_vec_col(target.to_vec());
        self.inner.fit_raw(&input, target, epochs);
    }

    #[wasm_bindgen]
    pub fn choice_bool(&self, input: Vec<f32>) -> bool {
        let input = Tensor::from_vec(input);
        self.inner.choice_bool(&input)
    }

    #[wasm_bindgen]
    pub fn choice(&self, input: Vec<f32>, options: Vec<JsValue>) -> JsValue {
        let input = Tensor::from_vec(input);
        let options_array = [options.get(0).cloned().unwrap_or(JsValue::null()), options.get(1).cloned().unwrap_or(JsValue::null())];
        self.inner.choice(&input, &options_array)
    }

    #[wasm_bindgen]
    pub fn predict_prob(&self, input: Vec<f32>) -> Vec<f32> {
        let input = Tensor::from_vec(input);
        self.inner.predict_prob(&input).to_vec()
    }


}

