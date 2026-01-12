use crate::data::scaler::StandardScaler;
use crate::nn::loss::{self, LOSS};
use crate::nn::model::Trainable;
use crate::core::tensor::Tensor;

use super::layer::Layer;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    loss: fn(&Tensor, &Tensor) -> f32
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, loss: LOSS) -> Self{
        let loss = loss::get_function(loss);
        Self {
            layers,
            loss
        }
    }
}

/* pub struct NeuralNetworkModel {
    network: NeuralNetwork,
    scaler: StandardScaler
} */

impl Trainable for NeuralNetwork {
    fn predict(&self, input: &Tensor) -> Tensor {
        assert!(self.layers[0].compatible(input), "The input shape is not compatible");
        let mut result = input.clone();
        for layer in self.layers.iter() {
            result = layer.forward(&result);
        }
        result
    }

    fn cost(&self, input: &Tensor, target: &Tensor) -> f32 {
        let result = self.predict(input);
        (self.loss)(&result, target)
    }

    fn gradient(&self, input: &Tensor, target: &Tensor) -> Tensor {
        Tensor::from_elem(1.0)
    }

    fn train_step(&mut self, input: &Tensor, target: &Tensor) {
        todo!()
    }
}