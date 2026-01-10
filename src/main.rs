use neural_network::core::tensor::Tensor;
use neural_network::nn::layer;


fn main() {
    let layer = layer::Layer::new((2, 1), neural_network::nn::activation::ACTIVATIONS::DEFAULT);
    println!("{}", layer);
    println!("{}", layer.forward(&Tensor::from_vec_col(vec![1.0, 2.0])));
}
