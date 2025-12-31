use neural_network::data;
use neural_network::nn::{perceptron::Perceptron};
use neural_network::nn::model::{Model, Trainable};

fn main () {
    //Raw data: lr - 0.012, epochs - 70
    //Normal data: lr - 1.1 epochs - 20
    let mut model = Perceptron::new(1, 1.0);
    let (inputs, data) = ([vec![-20.0], vec![-10.0], vec![0.0], vec![10.0], vec![20.0]], [-4.0, 14.0, 32.0, 50.0 ,68.0]);
    let scale = data::scaler::StandardScale::new(&inputs);
    let inputs = scale.transform_batch(&inputs);
    println!("{:?}", inputs);
    model.fit(&inputs, &data, 15);
    let result = model.evaluate(&inputs, &data);
    println!("{:?}", result);
    println!("{:?}", scale.transform(model.weights[0]))

}