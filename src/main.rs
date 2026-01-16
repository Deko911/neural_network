use neural_network::core::tensor::Tensor;
use neural_network::nn::layer::Layer;
use neural_network::nn::model::Trainable;
use neural_network::nn::neural_network::NeuralNetwork;


fn main() {
    let mut nn = NeuralNetwork::new(vec![
        Layer::new((3, 3), neural_network::nn::activation::ACTIVATIONS::SIGMOID),
        Layer::new((3, 3), neural_network::nn::activation::ACTIVATIONS::SIGMOID),
        Layer::new((3, 1), neural_network::nn::activation::ACTIVATIONS::SIGMOID)
        ], 0.1, neural_network::nn::loss::LOSS::BINARY_CROSS_ENTROPY);
    let student = Tensor::from_vec([9.0, 95.0, 90.0].to_vec());
    println!("{}", nn.predict(&student));
    let inputs = Tensor::from_matrix(
        &vec![
            [2.0, 60.0, 50.0].to_vec(),
            [4.0, 70.0, 65.0].to_vec(),
            [6.0, 80.0, 70.0].to_vec(),
            [8.0, 90.0, 85.0].to_vec(),
            [1.0, 50.0, 40.0].to_vec(),
            [3.0, 65.0, 55.0].to_vec(),
            [7.0, 85.0, 80.0].to_vec(),
            [9.0, 95.0, 90.0].to_vec(),
        ],
    );
    let targets = Tensor::from_vec_col(vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    for _ in 0..5000 {
        nn.train_step(&inputs, &targets);
    }
    println!("{}", nn.predict(&student));
}