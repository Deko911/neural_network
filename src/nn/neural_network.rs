use serde::{Deserialize, Serialize};

use crate::data::scaler::StandardScaler;
use crate::nn::activation::ACTIVATIONS;
use crate::nn::layer::INITIALIZER;
use super::loss::LOSS;
use super::model::{Metrics, Model, Trainable};
use crate::core::tensor::Tensor;

use super::layer::Layer;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Write};

const GRADIENT_THRESHOLD: f32 = 25.0;
const MIN_LEARNING: f32 = 0.00001;
const MIN_LR: f32 = 0.1;
const PATIENCE: usize = 5;
const LR_REDUCING_FACTOR: f32 = 0.8;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    lr: f32,
    loss: fn(&Tensor, &Tensor) -> f32,
    loss_name: LOSS
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, lr: f32, loss_name: LOSS) -> Self{
        let loss = LOSS::get_function(loss_name);
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
        let activation_prime = ACTIVATIONS::get_prime(last_activation);
        let cost_prime = LOSS::get_prime(self.loss_name);
        // delta = dC/da * da/dz
        let mut delta = if let ACTIVATIONS::SOFTMAX = last_activation {
            &activation - target
        }  else {
            cost_prime(&activation, target) * activation_prime(&zs.last().unwrap())
        };
        nabla_b[ls - 1] = delta.clone();
        nabla_w[ls - 1] = activations[ls - 1].t().dot(&delta);

        for i in (0..ls-1).rev() {
            let z = &zs[i];
            let activation_layer = &self.layers[i];
            let activation_prime = ACTIVATIONS::get_prime(activation_layer.activation_name);
            // delta = (delta * W^T) .* f'(z)
            delta = delta.dot(&self.layers[i+1].weights.t()) * activation_prime(z);
            nabla_b[i] = delta.clone();
            nabla_w[i] = activations[i].t().dot(&delta);
        }
        (nabla_w, nabla_b)
    }
}

pub struct NeuralNetworkModel {
    pub network: NeuralNetwork,
    input_scaler: StandardScaler,
    output_scaler: StandardScaler
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetworkSave {
    weights: Vec<Vec<f32>>,
    bias: Vec<Vec<f32>>,
    layers: Vec<usize>,
    initializers: Vec<INITIALIZER>,
    activations: Vec<ACTIVATIONS>,
    loss: LOSS,
    input_scaler: (Vec<f32>, Vec<f32>),
    output_scaler: (Vec<f32>, Vec<f32>)
}

impl NeuralNetworkModel {
    pub fn new(layers: Vec<Layer>, lr: f32, loss: Option<LOSS>) -> Self{
        let loss = loss.unwrap_or_default();
        Self {
            network: NeuralNetwork::new(layers, lr, loss),
            input_scaler: StandardScaler::new(),
            output_scaler: StandardScaler::new(),
        }
    }

    pub fn from_save(save: NeuralNetworkSave) -> Self {
        let NeuralNetworkSave { weights, bias, layers, activations, loss, input_scaler, output_scaler, initializers } = save;
        
        let mut last_size = layers[0];
        let mut parsed_layers: Vec<Layer> = vec![];
        for i in 1..layers.len() {
            let size = layers[i];
            let mut layer = Layer::new((last_size, size), activations[i - 1], initializers[i - 1]);
            layer.weights = Tensor::from_shape_vec((last_size, size), weights[i - 1].clone());
            layer.bias = Tensor::from_vec(bias[i - 1].clone());
            parsed_layers.push(layer);
            last_size = size;
        }
        
        let network = NeuralNetwork::new(parsed_layers, 1.0, loss);
        let input_scaler = StandardScaler::from_data(input_scaler);
        let output_scaler = StandardScaler::from_data(output_scaler);
        Self { network, input_scaler, output_scaler }
    }

    pub fn load_model(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let save: NeuralNetworkSave = serde_json::from_reader(reader)?;
        Ok(Self::from_save(save))
    }

    pub fn save(&self, path: &str) {
        let path = std::path::Path::new(path);
    
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        
        let mut file = File::create(path).unwrap();
        
        let mut weights: Vec<Vec<f32>> = vec![];
        let mut bias: Vec<Vec<f32>> = vec![];
        let mut activations: Vec<ACTIVATIONS> = vec![];
        let mut layers= vec![self.network.layers[0].weights.shape()[0]]; 
        let mut initializers: Vec<INITIALIZER> = vec![];

        for layer in self.network.layers.iter() {
            layers.push(layer.weights.shape()[1]);
            weights.push(layer.weights.to_vec());
            bias.push(layer.bias.to_vec());
            activations.push(layer.activation_name);
            initializers.push(layer.initializer_name);
        }

        let data = NeuralNetworkSave {
            weights, 
            bias,
            layers,
            activations,
            initializers,
            loss: self.network.loss_name,
            input_scaler: self.input_scaler.get_data(),
            output_scaler: self.output_scaler.get_data()
        };
        let json_string = serde_json::to_string_pretty(&data).unwrap();
        file.write(json_string.as_bytes()).unwrap();
    }

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

    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32 {
        let mut cost = 0.0;
        let n = input.len();
        // Gradient for the batch
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
        // Update weighs and bias
        for l in 0..self.layers.len() {
            let mut grad_w = nabla_w_sum[l].clone().unwrap();
            let mut grad_b = nabla_b_sum[l].clone().unwrap();

            if grad_w.norm() > GRADIENT_THRESHOLD {
                let norm = GRADIENT_THRESHOLD / grad_w.norm();
                grad_w = grad_w * norm
            }

            if grad_b.norm() > GRADIENT_THRESHOLD {
                let norm = GRADIENT_THRESHOLD / grad_b.norm();
                grad_b = grad_b * norm
            }

            let avg_dw = grad_w * (self.lr / n as f32);
            let avg_db = grad_b * (self.lr / n as f32);
            self.layers[l].weights = &self.layers[l].weights - &avg_dw;
            self.layers[l].bias = &self.layers[l].bias - &avg_db;
        }
        println!("cost {}", cost);
        return cost;
    }
}

impl Model for NeuralNetworkModel {
    
    fn predict(&self, input: &Tensor) -> Tensor {
        let input = self.input_scaler.transform(input);
        let result = self.network.predict(&input);
        self.output_scaler.inverse_transform(&result)
    }
    
    fn fit_raw(&mut self, input: &Tensor, target: &Tensor, epochs: usize, batch_size: usize) {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        let batch_size = if batch_size == 0 { input.size() } else { batch_size };
        let mut last_cost = f32::NAN;
        for i in 0..epochs {
            let batches = Tensor::create_batches(&input, &target, batch_size);
            let mut cost = 0.0;
            let n_batches = batches.len();
            for (batch_i, batch_t) in batches {
                cost += self.network.train_step(&batch_i, &batch_t) / n_batches as f32;
            }
            if last_cost.is_nan(){
                last_cost = cost;
                continue;
            }
            if i % PATIENCE == 0 {
                if last_cost - cost < MIN_LEARNING {
                    self.network.lr = MIN_LR.max(self.network.lr * LR_REDUCING_FACTOR)
                }
            }
            last_cost = cost
        }
    }
    
    fn fit(&mut self, input: &Tensor, target: &Tensor, epochs: usize, batch_size: usize) {
        assert_eq!(
            input.len(),
            target.len(),
            "There must be as many inputs as targets."
        );
        let batch_size = if batch_size == 0 { input.size() } else { batch_size };
        self.input_scaler.fit(input);
        let input = self.input_scaler.transform(input);
        let mut target = target.clone();
        if LOSS::is_scalable(self.network.loss_name){
            self.output_scaler.fit(&target);
            target = self.output_scaler.transform(&target);
        }
        let mut last_cost = f32::NAN;
        for i in 0..epochs {
            let batches = Tensor::create_batches(&input, &target, batch_size);
            let mut cost = 0.0;
            let n_batches = batches.len();
            for (batch_i, batch_t) in batches {
                cost += self.network.train_step(&batch_i, &batch_t) / n_batches as f32;
            }
            if last_cost.is_nan(){
                last_cost = cost;
                continue;
            }
            if i % PATIENCE == 0 {
                if last_cost - cost < MIN_LEARNING {
                    self.network.lr = MIN_LR.max(self.network.lr * LR_REDUCING_FACTOR)
                }
            }
            last_cost = cost
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