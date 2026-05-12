use neural_network::core::tensor::Tensor;
use neural_network::nn::layer::Layer;
use neural_network::nn::model::{Metrics, Model};
use neural_network::nn::neural_network::NeuralNetworkModel;
use neural_network::nn::activation::ACTIVATIONS;
use neural_network::nn::layer::INITIALIZER;

fn main() {
    let mut nn = NeuralNetworkModel::new(vec![
        Layer::new((2, 8), ACTIVATIONS::RELU, INITIALIZER::HE),
        Layer::new((8, 4), ACTIVATIONS::RELU, INITIALIZER::HE),
        Layer::new((4, 1), ACTIVATIONS::SIGMOID, INITIALIZER::XAVIER)
        ], 5.0, Some(neural_network::nn::loss::LOSS::BINARY_CROSS_ENTROPY));
    let student = Tensor::from_vec([1.0, 1.0].to_vec());
    let inputs = Tensor::from_matrix(
        &vec![
            [0.0, 0.0].to_vec(),
            [1.0, 0.0].to_vec(),
            [0.0, 1.0].to_vec(),
            [1.0, 1.0].to_vec()
        ],
    );
    let targets = Tensor::from_vec_col(vec![0.0, 1.0, 1.0, 0.0]);
    nn.fit(&inputs, &targets, 3000);
    println!("{}", nn.evaluate(&inputs, &targets));
    println!("{}", nn.predict(&student));
}