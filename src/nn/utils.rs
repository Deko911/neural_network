use crate::core::tensor::Tensor;

pub fn vec_to_input (vec: Vec<f32>) -> Tensor {
    Tensor::from_shape_vec((1, vec.len()), vec)
}