use neural_network::core::tensor::Tensor;
use neural_network::nn::layer::Layer;
use neural_network::nn::model::{Metrics, Model};
use neural_network::nn::neural_network::NeuralNetworkModel;

fn main() {
    // [size (m2), rooms, years]
    let inputs = Tensor::from_matrix(&vec![
        vec![50.0, 2.0, 10.0],
        vec![80.0, 3.0, 5.0],
        vec![60.0, 2.0, 20.0],
        vec![120.0, 4.0, 2.0],
        vec![100.0, 3.0, 15.0],
        vec![70.0, 2.0, 7.0],
        vec![90.0, 3.0, 12.0],
        vec![110.0, 4.0, 3.0],
    ]);
    // Price as thousands
    let targets = Tensor::from_vec_col(vec![120.0, 200.0, 110.0, 300.0, 220.0, 150.0, 210.0, 280.0]);

    let mut nn = NeuralNetworkModel::new(
        vec![
            Layer::new((3, 8), neural_network::nn::activation::ACTIVATIONS::LINEAR),
            Layer::new((8, 4), neural_network::nn::activation::ACTIVATIONS::LINEAR),
            Layer::new((4, 1), neural_network::nn::activation::ACTIVATIONS::LINEAR),
        ],
        0.07,
        Some(neural_network::nn::loss::LOSS::QUAD),
    );

    nn.fit(&inputs, &targets, 50);
    println!("Cost {}", nn.evaluate(&inputs, &targets));

    let new_house = Tensor::from_vec(vec![95.0, 3.0, 8.0]);
    let predicted_price = nn.predict(&new_house);
    println!("Price {:.2}", predicted_price.as_f32());
}