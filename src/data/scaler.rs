use crate::core::tensor::Tensor;

pub struct StandardScaler {
    mean: Tensor,
    std: Tensor,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: Tensor::from_elem(0.0),
            std: Tensor::from_elem(1.0),
        }
    }

    pub fn fit(&mut self, data: &Tensor) {
        let data = data.t();
        let n = data.len();
        let mut mean_vec: Vec<f32> = vec![];
        let mut std_vec: Vec<f32> = vec![];

        for i in 0..n {
            let data_slice = data.row(i);
            let size = data_slice.size() as f32;
            let mean = data_slice.iter().map(|x| x).sum::<f32>() / size;
            let sum = data_slice.iter().map(|x| (*x - mean).powi(2)).sum::<f32>() / size;
            let std = sum.sqrt().max(1e-8);
            mean_vec.push(mean);
            std_vec.push(std);
        }

        self.mean = Tensor::from_vec(mean_vec);
        self.std = Tensor::from_vec(std_vec);
    }

    pub fn transform(&self, input: &Tensor) -> Tensor {
        &(input - &self.mean) / &self.std
    }

    pub fn inverse_transform(&self, x: &Tensor) -> Tensor {
        &self.std * x + &self.mean
    }
}
