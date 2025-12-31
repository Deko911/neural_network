use crate::core::linalg;

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

    pub fn predict(&self, input: &[f32]) -> i32 {
        let z = linalg::dot(&self.weights, input);
        if z >= 0.0 { 1 } else { 0 }
    }

    pub fn train(&mut self, input: &[f32], target: i32) {
        let prediction = self.predict(input);
        let error = target - prediction;
        
        for i in 0..self.weights.len() {
            println!("{}", self.lr * error as f32 * input[i]);
            self.weights[i] += self.lr * error as f32 * input[i]
        }
        println!("{:?}", self.weights);

        self.bias += self.lr * error as f32;
    }
}

pub fn fit(
    model: &mut Perceptron,
    x: &[Vec<f32>],
    y: &[i32],
    epochs: usize,
) {
    for _ in 0..epochs {
        for (input, target) in x.iter().zip(y) {
            model.train(input, *target);
        }
    }
}