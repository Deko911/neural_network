use crate::core::tensor::Tensor;

pub trait Model {
    
    fn predict(&self, input: &Tensor) -> Tensor;
    fn fit(&mut self, input: &Tensor, target: &Tensor, epochs: usize);
    fn fit_raw(&mut self, input: &Tensor, target: &Tensor, epochs: usize);
}

pub trait Trainable {
    fn predict(&self, input: &Tensor) -> Tensor;
    fn cost(&self, input: &Tensor, target: &Tensor) -> f32;
    fn gradient(&self, input: &Tensor, target: &Tensor) -> Tensor;
    fn train_step(&mut self, input: &Tensor, target: &Tensor);
}

pub trait Metrics {
    fn evaluate(&self, input: &Tensor, target: &Tensor) -> f32;
    fn accurate(&self, input: &Tensor, target: &Tensor) -> f32;
}
