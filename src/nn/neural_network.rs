use crate::data::scaler::StandardScaler;
use crate::nn::activation;
use super::loss::{self, LOSS};
use super::model::{Metrics, Model, Trainable};
use crate::core::tensor::Tensor;

use super::layer::Layer;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    lr: f32,
    loss: fn(&Tensor, &Tensor) -> f32,
    loss_name: LOSS
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, lr: f32, loss_name: LOSS) -> Self{
        let loss = loss::get_function(loss_name);
        Self {
            layers,
            lr,
            loss,
            loss_name
        }
    }


    //(w, b)
    pub fn backpropagation(&self, input: &Tensor, target: &Tensor) -> (Vec<Tensor>, Vec<Tensor>) {
        let ls = self.layers.len();
        let mut zs: Vec<Tensor> = vec![];
        let mut activations: Vec<Tensor> = vec![input.clone()];
        let mut activation: Tensor = input.clone(); //This is outdated by one position
        let mut nabla_w = vec![Tensor::zeros((1,1)); ls];
        let mut nabla_b = vec![Tensor::zeros((1,1)); ls];
        for layer in self.layers.iter() {
            let z: Tensor;
            (z, activation) = layer.forward(&activation);
            zs.push(z.clone());
            activations.push(activation.clone());
        }
        let last_activation = self.layers.last().unwrap().activation_name;
        let activation_prime = activation::get_prime(last_activation);
        let cost_prime = loss::get_prime(self.loss_name);
        // delta = dC/da * da/dz
        let mut delta = cost_prime(&activation, target) * activation_prime(&zs.last().unwrap());
        nabla_b[ls - 1] = delta.clone();
        nabla_w[ls - 1] = activations[ls - 1].t().dot(&delta);

        for i in (0..ls-1).rev() {
            let z = &zs[i];
            let activation_layer = &self.layers[i];
            let activation_prime = activation::get_prime(activation_layer.activation_name);
            // delta = (delta * W^T) .* f'(z)
            delta = delta.dot(&self.layers[i+1].weights.t()) * activation_prime(z);
            nabla_b[i] = delta.clone();
            nabla_w[i] = activations[i].t().dot(&delta);
        }
        (nabla_w, nabla_b)
    }
}

pub struct NeuralNetworkModel {
    network: NeuralNetwork,
    scaler: StandardScaler
}

impl Trainable for NeuralNetwork {
    fn predict(&self, input: &Tensor) -> Tensor {
        assert!(self.layers[0].compatible(input), "The input shape is not compatible");
        let mut result = input.clone();
        for layer in self.layers.iter() {
            (_, result) = layer.forward(&result);
        }
        result
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

    fn train_step(&mut self, input: &Tensor, target: &Tensor) {
        let mut cost = 0.0;
        let n = input.len();
        // Acumular gradientes para batch
        let mut nabla_w_sum: Vec<Option<Tensor>> = vec![None; self.layers.len()];
        let mut nabla_b_sum: Vec<Option<Tensor>> = vec![None; self.layers.len()];
        for i in 0..n {
            let input_slice = input.row(i);
            let target_slice = target.row(i);
            let (delta_weights, delta_bias) = self.backpropagation(&input_slice, &target_slice);
            let result = self.predict(&input_slice);
            cost += self.cost(&result, &target_slice) / n as f32;
            for l in 0..self.layers.len() {
                nabla_w_sum[l] = Some(match &nabla_w_sum[l] {
                    Some(ref acc) => acc + &delta_weights[l],
                    None => delta_weights[l].clone(),
                });
                nabla_b_sum[l] = Some(match &nabla_b_sum[l] {
                    Some(ref acc) => acc + &delta_bias[l],
                    None => delta_bias[l].clone(),
                });
            }
        }
        // Actualizar pesos y bias con el gradiente promedio
        for l in 0..self.layers.len() {
            let avg_dw = nabla_w_sum[l].as_ref().unwrap() * (self.lr / n as f32);
            let avg_db = nabla_b_sum[l].as_ref().unwrap() * (self.lr / n as f32);
            self.layers[l].weights = &self.layers[l].weights - &avg_dw;
            self.layers[l].bias = &self.layers[l].bias - &avg_db;
        }
        println!("cost {}", cost);
    }
}

impl Model for NeuralNetworkModel {
    
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

impl Metrics for NeuralNetworkModel {
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