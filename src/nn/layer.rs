//XOR Perceptron - Sigmoid: Accurate - 0.5, Evaluation - 0.89, Epochs - 150

use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

use crate::core::tensor::Tensor;
use super::activation::ACTIVATIONS;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum INITIALIZER {
    XAVIER,
    HE
}

impl INITIALIZER {
    pub fn init(initializer: INITIALIZER, fan_in: usize, fan_out: usize) -> Tensor {
        use INITIALIZER::*;
        match initializer {
            XAVIER => Tensor::xavier_init(fan_in, fan_out),
            HE => Tensor::he_init(fan_in, fan_out)
        }
    }
}

pub struct Layer {
    pub weights: Tensor,
    pub bias: Tensor,
    activation: fn(&Tensor) -> Tensor,
    pub activation_name: ACTIVATIONS,
    pub initializer_name: INITIALIZER
}

impl Layer {
    ///
    /// (inputs, neurons)
    pub fn new(shape: (usize, usize), activation: ACTIVATIONS, initializer: INITIALIZER) -> Self {
        let weights = INITIALIZER::init(initializer, shape.0, shape.1);
        let bias = Tensor::zeros((1, shape.1));
        let activation_name = activation;
        let activation = ACTIVATIONS::get_function(activation);
        Self { weights, bias, activation, activation_name, initializer_name: initializer }
    }

    //(z, activation)
    pub fn forward(&self, input: &Tensor) -> (Tensor, Tensor) {
        let x = &input.dot(&self.weights) + &self.bias;
        let activation = (self.activation)(&x);
        (x, activation)
    }

    pub fn compatible(&self, rhs: &Tensor) -> bool {
        let (rows1, _) = self.weights.shape_tuple();
        let (rows2, cols2) = rhs.shape_tuple();
        rows1 == cols2 && rows2 == 1
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer: \n Size {:?} \n Weights \n {} \n Bias \n {} \n Activation {:?}", self.weights.shape_tuple(), self.weights, self.bias, self.activation_name)
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer({} + {}) -> {:?}", self.weights, self.bias, self.activation_name)
    }
}