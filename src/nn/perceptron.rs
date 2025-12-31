use super::model::Model;
use crate::{core::linalg, nn::model::Trainable};

pub struct Perceptron {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub lr: f32,
}

impl Perceptron {
    pub fn new(input_size: usize, lr: f32) -> Self {
        Self {
            weights: vec![0.0; input_size],
            bias: 0.0,
            lr,
        }
    }
    fn error(&self, input: &Vec<f32>, target: &f32) -> f32 {
        let result = self.predict(input);
        target - result
    }
}

impl Model for Perceptron {
    type Input = Vec<f32>;
    type Output = f32;
    fn predict(&self, input: &Self::Input) -> Self::Output {
        linalg::dot(&self.weights, input) + self.bias
    }

    fn evaluate(&self, input: &[Self::Input], target: &[Self::Output]) -> f32 {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        let mut total = 0.0;
        for i in 0..input.len() {
            let result = self.predict(&input[i]);
            let error = target[i] - result;
            total += error  / (if target[i] == 0.0 {1.0} else {target[i]});
        }
        total / input.len() as f32
    }

    fn evaluate_one(&self, input: &Self::Input, target: &Self::Output) -> f32 {
        let result = self.predict(input);
        (target - result) / (if target == &0.0 {1.0} else {*target})
    }
}

impl Trainable for Perceptron {
    fn train_step(&mut self, input: &[Self::Input], target: &[Self::Output]) {
        for i in 0..input.len() {
            let error = self.error(&input[i], &target[i]);
            for j in 0..self.weights.len() {
                self.weights[j] += self.lr * error * input[i][j]
            }
            self.bias += self.lr * error;
            println!("{:?} {:?}, {:?}", self.weights, self.bias, error);
        }
    }
}
