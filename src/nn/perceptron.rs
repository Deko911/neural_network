use super::model::{Model, Trainable};
use crate::core::tensor::Tensor;
use crate::data::scaler::StandardScaler;
use crate::nn::activation::ACTIVATIONS;
use crate::nn::loss::LOSS;
use crate::nn::model::Metrics;

pub struct Perceptron {
    weights: Tensor,
    bias: f32,
    lr: f32,
    activation: fn(&Tensor) -> Tensor,
    loss: fn(&Tensor, &Tensor) -> f32,
    loss_name: LOSS
}

pub struct PerceptronModel {
    network: Perceptron,
    input_scaler: StandardScaler,
    output_scaler: StandardScaler
}

impl Perceptron {
    pub fn new(input_size: usize, lr: f32, activation: ACTIVATIONS, loss_name: LOSS) -> Self {
        let activation = ACTIVATIONS::get_function(activation);
        let loss = LOSS::get_function(loss_name);
        Self {
            weights: Tensor::zeros((input_size, 1)),
            bias: 0.0,
            lr,
            activation,
            loss,
            loss_name
        }
    }
}

impl PerceptronModel {
    pub fn new(input_size: usize, lr: f32, activation: Option<ACTIVATIONS>, loss: Option<LOSS>) -> Self {
        let activation = activation.unwrap_or_default();
        let loss = loss.unwrap_or_default();
        Self {
            network: Perceptron::new(input_size, lr, activation, loss),
            input_scaler: StandardScaler::new(),
            output_scaler: StandardScaler::new()
        }
    }
}

impl Trainable for Perceptron {

    fn predict(&self, input: &Tensor) -> Tensor {
        let z = input.dot(&self.weights) + self.bias;
        (self.activation)(&z)
    }
    
    fn cost(&self, result: &Tensor, target: &Tensor) -> f32 {
        (self.loss)(&result, target)
    }

    fn gradient(&self, cost: f32, result: &Tensor, target: &Tensor) -> Tensor {
        let target_slice = target.as_slice().unwrap();
        let mut idx = 0;
        result.map(|x| {
            let err = (target_slice[idx] - x).signum();
            idx += 1;
            err * cost
        })
    }
    
    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32{
        let mut cost_avr = 0.0;
        for i in 0..input.len() {
            let input_slice = input.row(i);
            let target_slice = target.row(i);
            let result = self.predict(&input_slice);
            let cost = self.cost(&result, &target_slice);
            let error = self.gradient(cost, &result, &target_slice);
            self.weights += &input_slice.t() * &error * self.lr;
            self.bias += error.as_f32() * self.lr;
            cost_avr += cost / input.len() as f32;
        }
        return  cost_avr;
    }
}

impl Model for PerceptronModel {
    
    fn predict(&self, input: &Tensor) -> Tensor {
        let input = self.input_scaler.transform(input);
        let result = self.network.predict(&input);
        self.output_scaler.inverse_transform(&result)
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
        self.input_scaler.fit(input);
        let input = self.input_scaler.transform(input);
        let mut target = target.clone();
        if LOSS::is_scalable(self.network.loss_name){
            self.output_scaler.fit(&target);
            target = self.output_scaler.transform(&target);
        }
        for _ in 0..epochs {
            self.network.train_step(&input, &target);
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
        let mut total = 0.0;
        for i in 0..input.len() {
            let input_slice = input.row(i);
            let target_slice = target.row(i);
            let result = self.predict(&input_slice);
            let error = self.network.cost(&result, &target_slice);
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