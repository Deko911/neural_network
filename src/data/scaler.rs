use crate::core::tensor::Tensor;

pub struct StandardScaler {
    mean: f32,
    std: f32,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
        }
    }

    pub fn fit(&mut self, data: &Tensor) {
        let mean = data.iter().map(|x| x).sum::<f32>() / data.size() as f32;
        let sum = data.iter().map(|x| (*x - mean).powi(2)).sum::<f32>() / data.size() as f32;
        let std = sum.sqrt().max(1e-8);
        self.mean = mean;
        self.std = std;
    }

    pub fn transform(&self, input: &Tensor) -> Tensor {
        let data: Vec<f32> = input.iter().map(|x| (*x - self.mean) / self.std).collect();
        let (rows, cols) = input.shape_tuple();
        let tensor = Tensor::from_shape_vec((rows, cols), data);
        tensor
    }

    pub fn inverse_transform(&self, x: f32) -> f32 {
        x * self.std + self.mean
    }
}
