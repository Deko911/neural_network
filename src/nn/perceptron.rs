use super::model::{Model, Trainable};
use crate::core::tensor::Tensor;
use crate::data::scaler::StandardScaler;
use crate::nn::activation::{self, ACTIVATIONS};
use crate::nn::loss::{self, LOSS};
use crate::nn::model::Metrics;

pub struct Perceptron {
    weights: Tensor,
    bias: f32,
    lr: f32,
    activation: fn(Tensor) -> Tensor,
    loss: fn(&Tensor, &Tensor) -> f32
}

pub struct PerceptronModel {
    network: Perceptron,
    scaler: StandardScaler,
}

impl Perceptron {
    pub fn new(input_size: usize, lr: f32, activation: ACTIVATIONS, loss: LOSS) -> Self {
        let activation = activation::get_function(activation);
        let loss = loss::get_function(loss);
        Self {
            weights: Tensor::zeros((input_size, 1)),
            bias: 0.0,
            lr,
            activation,
            loss
        }
    }
}

impl PerceptronModel {
    pub fn new(input_size: usize, lr: f32, activation: Option<ACTIVATIONS>, loss: Option<LOSS>) -> Self {
        let activation = activation.unwrap_or_default();
        let loss = loss.unwrap_or_default();
        Self {
            network: Perceptron::new(input_size, lr, activation, loss),
            scaler: StandardScaler::new(),
        }
    }
}

impl Trainable for Perceptron {

    fn predict(&self, input: &Tensor) -> Tensor {
        let z = input.dot(&self.weights) + self.bias;
        (self.activation)(z)
    }
    
    fn cost(&self, input: &Tensor, target: &Tensor) -> f32 {
        let result = self.predict(input);
        (self.loss)(&result, target)
    }

    fn gradient(&self, input: &Tensor, target: &Tensor) -> Tensor {
        let result = self.predict(input);
        let cost = (self.loss)(&result, target);
        let target_slice = target.as_slice().unwrap();
        let mut idx = 0;
        result.map(|x| {
            let err = (target_slice[idx] - x).signum();
            idx += 1;
            err * cost
        })
    }
    
    fn train_step(&mut self, input: &Tensor, target: &Tensor) {
        for i in 0..input.len() {
            let input_slice = input.row(i);
            let target_slice = target.row(i);
            let error = self.gradient(&input_slice, &target_slice);
            self.weights += &input_slice.t() * &error * self.lr;
            self.bias += error.as_f32() * self.lr;
        }
    }
}

impl Model for PerceptronModel {
    
    fn predict(&self, input: &Tensor) -> Tensor {
        let input = self.scaler.transform(input);
        self.network.predict(&input)
    }
    
    fn fit_raw(&mut self, input: &Tensor, target: &Tensor, epochs: usize) {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        for _ in 0..epochs {
            self.network.train_step(&input, target);
        }
    }
    
    fn fit(&mut self, input: &Tensor, target: &Tensor, epochs: usize) {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        self.scaler.fit(input);
        let input = self.scaler.transform(input);
        for _ in 0..epochs {
            self.network.train_step(&input, target);
        }
    }
}

impl Metrics for PerceptronModel {
    fn evaluate(&self, input: &Tensor, target: &Tensor) -> f32 {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        let input = self.scaler.transform(input);
        let mut total = 0.0;
        for i in 0..input.len() {
            let input_slice = input.row(i);
            let target_slice = target.row(i);
            let error = 1.0 / (self.network.cost(&input_slice, &target_slice) + 1.0);
            total += error;
        }
        total / input.len() as f32
    }

    fn accurate(&self, input: &Tensor, target: &Tensor) -> f32 {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        let mut total = 0.0;
        for i in 0..input.len() {
            let input_slice = input.row(i);
            let target_slice = target.row(i);
            let targetf32 = target_slice.as_f32();
            let result = self.predict(&input_slice).as_f32();
            let error = (targetf32 - result).abs();
            if targetf32 > error {
                total += error / (if targetf32 == 0.0 { 1.0 } else { targetf32.abs() });
            }else {
                total += error / (if result == 0.0 { 1.0 } else { result.abs() }); 
            }
        }
        1.0 - total / input.len() as f32
    }
}