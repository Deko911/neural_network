extern crate neural_network;
use neural_network::nn::perceptron;

fn main() {
    let mut model = perceptron::Perceptron::new(2, 0.1);
    println!("{}", model.predict(&[0.0, 0.0]));
    perceptron::fit(&mut model, &[vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]], &[0, 0, 0, 1], 100);
    println!("{}", model.predict(&[0.0, 0.0]));
    println!("{:?}", model.weights);
}