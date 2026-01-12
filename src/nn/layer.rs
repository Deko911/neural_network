//XOR Perceptron - Sigmoid: Accurate - 0.5, Evaluation - 0.89, Epochs - 150

use std::fmt::{Debug, Display};

use crate::core::tensor::Tensor;
use super::activation::{self, ACTIVATIONS};

pub struct Layer {
    weights: Tensor,
    bias: Tensor,
    activation: fn(Tensor) -> Tensor,
    activation_name: ACTIVATIONS
}

impl Layer {
    ///
    /// (inputs, neurons)
    pub fn new(shape: (usize, usize), activation: ACTIVATIONS) -> Self {
        let weights = Tensor::random(shape);
        let bias = Tensor::zeros((1, shape.1));
        let activation_name = activation;
        let activation = activation::get_function(activation);
        Self { weights, bias, activation, activation_name }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let x = &input.dot(&self.weights) + &self.bias;
        (self.activation)(x)
    }

    pub fn compatible(&self, rhs: &Tensor) -> bool {
        let (rows1, cols1) = self.weights.shape_tuple();
        let (rows2, cols2) = rhs.shape_tuple();
        rows1 == cols2 && cols1 == rows2
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