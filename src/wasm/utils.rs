use crate::core::tensor::Tensor;

pub fn parse_data_to_tensor(data_size: usize, data: Vec<f32>) -> Tensor {
    let tensor = Tensor::from_shape_vec((data.len() / data_size, data_size), data);
    tensor
}