use super::model::{Model, Trainable};
use super::perceptron::Perceptron;
use crate::core::tensor::Tensor;
use crate::data::scaler::StandardScaler;
use crate::nn::activation::ACTIVATIONS;

pub struct LogisticRegression {
    network: Perceptron,
    scaler: StandardScaler,
    threshold: f32,
}

impl LogisticRegression {
    pub fn new(input_size: usize, lr: f32, threshold: Option<f32>) -> Self {
        let threshold = threshold.unwrap_or_else(|| 0.5);
        let activation = ACTIVATIONS::SIGMOID;
        Self {
            network: Perceptron::new(input_size, lr, activation),
            scaler: StandardScaler::new(),
            threshold,
        }
    }

    pub fn choice_bool(&self, input: &Tensor) -> bool {
        let option = self.predict(input);
        option.as_f32() > self.threshold
    }

    pub fn choice<T: Clone>(&self, input: &Tensor, options: &[T; 2]) -> T {
        if self.choice_bool(input) {
            options[1].clone()
        } else {
            options[0].clone()
        }
    }

    pub fn predict_prob(&self, input: &Tensor) -> [f32; 2] {
        let result = self.predict(input).as_f32();
        [1.0 - result, result]
    }
}

impl Model for LogisticRegression {
    
    fn predict(&self, input: &Tensor) -> Tensor {
        let input = self.scaler.transform(input);
        self.network.predict(&input)
    }
    
    fn evaluate(&self, input: &Tensor, target: &Tensor) -> f32 {
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
            let result = self.choice(&input_slice, &[0.0, 1.0]);
            let error = targetf32 - result;
            total += error / (if targetf32 == 0.0 { 1.0 } else { targetf32 });
        }
        total / input.len() as f32
    }

    fn evaluate_one(&self, input: &Tensor, target: &Tensor) -> f32 {
        let target = target.as_f32();
        let result = self.choice(&input, &[0.0, 1.0]);
        (target - result) / (if target == 0.0 { 1.0 } else { target })
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


/* impl Model for LogisticRegression {
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
            let result = self.choice(&input[i], &[0.0, 1.0]);
            let error = (target[i] - result).abs();
            println!("{}, {}", target[i], result);
            total += error / (if target[i] == 0.0 { 1.0 } else { target[i] });
        }
        total / input.len() as f32
    }

    fn evaluate_one(&self, input: &Self::Input, target: &Self::Output) -> f32 {
        let result = self.choice(&input, &[0.0, 1.0]);
        (target - result).abs() / (if target == &0.0 { 1.0 } else { *target })
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
 */