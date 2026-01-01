use super::model::{Model, Trainable};
use crate::core::linalg;
use crate::data::scaler::StandardScale;

struct Perceptron {
    weights: Vec<f32>,
    bias: f32,
    lr: f32,
}

pub struct PerceptronModel {
    network: Perceptron,
    scaler: StandardScale,
}

impl Perceptron {
    fn new(input_size: usize, lr: f32) -> Self {
        Self {
            weights: vec![0.0; input_size],
            bias: 0.0,
            lr,
        }
    }
}

impl PerceptronModel {
    pub fn new(input_size: usize, lr: f32) -> Self {
        Self {
            network: Perceptron::new(input_size, lr),
            scaler: StandardScale::new(),
        }
    }
}

impl Trainable for Perceptron {
    type Input = Vec<f32>;
    type Output = f32;

    fn predict(&self, input: &Self::Input) -> Self::Output {
        linalg::dot(&self.weights, input) + self.bias
    }

    fn error(&self, input: &Self::Input, target: &Self::Output) -> f32 {
        let result = self.predict(input);
        target - result
    }

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

impl Model for PerceptronModel {
    type Input = Vec<f32>;
    type Output = f32;
    
    fn predict(&self, input: &Self::Input) -> Self::Output {
        let input = self.scaler.transform(input);
        self.network.predict(&input)
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
            total += error / (if target[i] == 0.0 { 1.0 } else { target[i] });
        }
        total / input.len() as f32
    }

    fn evaluate_one(&self, input: &Self::Input, target: &Self::Output) -> f32 {
        let result = self.predict(input);
        (target - result) / (if target == &0.0 { 1.0 } else { *target })
    }

    fn fit_raw(&mut self, input: &[Self::Input], target: &[Self::Output], epochs: usize) {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        for _ in 0..epochs {
            self.network.train_step(&input, target);
        }
    }

    fn fit(&mut self, input: &[Self::Input], target: &[Self::Output], epochs: usize) {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        self.scaler.fit(input);
        let input = self.scaler.transform_batch(input);
        for _ in 0..epochs {
            self.network.train_step(&input, target);
        }
    }
}
