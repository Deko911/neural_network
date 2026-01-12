use neural_network::core::tensor::Tensor;
use neural_network::nn::layer::Layer;
use neural_network::nn::model::Trainable;
use neural_network::nn::neural_network::NeuralNetwork;


fn main() {
    let nn = NeuralNetwork::new(vec![
        Layer::new((2, 1), neural_network::nn::activation::ACTIVATIONS::SIGMOID),
        Layer::new((1, 1), neural_network::nn::activation::ACTIVATIONS::SIGMOID)
    ], neural_network::nn::loss::LOSS::DEFAULT);
    println!("{}", nn.predict(&Tensor::from_vec(vec![1.0, 2.0])));
}
