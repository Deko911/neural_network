use crate::core::tensor::Tensor;

pub fn parse_array_to_tensor(input_size: usize, input: Vec<f32>) -> Tensor {
    let tensor = Tensor::from_shape_vec((input.len() / input_size, input_size), input);
    tensor
}
