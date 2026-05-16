use wasm_bindgen::prelude::*;

use crate::core::tensor::Tensor;
use crate::nn::activation::ACTIVATIONS; 
use crate::nn::layer::{INITIALIZER, Layer};
use crate::nn::loss::LOSS;
use crate::nn::model::*;
use crate::nn::neural_network::NeuralNetworkModel;
use crate::utils::set_panic_hook;
use super::utils::*;

#[wasm_bindgen]
pub struct NeuralNetworkJS {
    inner: NeuralNetworkModel,
    pub input_size: usize,
    pub output_size: usize
}

#[wasm_bindgen]
impl NeuralNetworkJS {
    #[wasm_bindgen(constructor)]
    pub fn new(input_size: usize, lr: f32, layer_sizes: Vec<usize>, activations: Vec<ACTIVATIONS>, initializers: Vec<INITIALIZER>, loss: Option<LOSS>) -> Self {
        set_panic_hook();
        assert!(layer_sizes.len() - 1 == activations.len() && layer_sizes.len() - 1 == initializers.len(), "The number of layers does not much with the data");
        let mut layers: Vec<Layer> = vec![];
        let mut last_size = layer_sizes[0];
        for i in 1..layer_sizes.len() {
            let size = layer_sizes[i];
            let a = activations[i - 1];
            let init = initializers[i - 1];
            layers.push(Layer::new((last_size, size), a, init));
            last_size = size
        }
        let inner = NeuralNetworkModel::new(layers, lr, loss);
        Self { inner, input_size, output_size: last_size }
    }
    
    #[wasm_bindgen]
    pub fn load(str: String) -> Option<Self> {
        let inner = NeuralNetworkModel::from_string(str)?;
        let input_size = inner.network.layers[0].weights.shape()[0];
        let output_size = inner.network.layers[inner.network.layers.len() - 1].weights.shape()[1];
        Some(NeuralNetworkJS { inner, input_size, output_size })
    }

    #[wasm_bindgen]
    pub fn save(&self) -> String {
        self.inner.to_string()
    }

    #[wasm_bindgen]
    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        let input = Tensor::from_vec(input);
        self.inner.predict(&input).to_vec()
    }

    #[wasm_bindgen]
    pub fn evaluate(&self, input: Vec<f32>, target: Vec<f32>) -> f32 {
        let input = parse_data_to_tensor(self.input_size, input);
        let target = parse_data_to_tensor(self.output_size, target);
        self.inner.evaluate(&input, &target)
    }

    #[wasm_bindgen]
    pub fn accurate(&self, input: Vec<f32>, target: Vec<f32>) -> f32 {
        let input = parse_data_to_tensor(self.input_size, input);
        let target = parse_data_to_tensor(self.output_size, target);
        self.inner.accurate(&input, &target)
    }

    #[wasm_bindgen]
    pub fn fit(&mut self, input: Vec<f32>, target: Vec<f32>, epochs: usize, batch_size: usize) {
        let input = parse_data_to_tensor(self.input_size, input);
        let target = parse_data_to_tensor(self.output_size, target);
        self.inner.fit(&input, &target, epochs, batch_size);
    }

    #[wasm_bindgen]
    pub fn fit_raw(&mut self, input: Vec<f32>, target: Vec<f32>, epochs: usize, batch_size: usize) {
        let input = parse_data_to_tensor(self.input_size, input);
        let target = parse_data_to_tensor(self.output_size, target);
        self.inner.fit_raw(&input, &target, epochs, batch_size);
    }

    #[wasm_bindgen]
    pub fn train_step(&mut self, input: Vec<f32>, target: Vec<f32>) -> f32 {
        let input = parse_data_to_tensor(self.input_size, input);
        let target = parse_data_to_tensor(self.output_size, target);
        return self.inner.network.train_step(&input, &target);
    }

}


