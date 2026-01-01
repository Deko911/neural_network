use neural_network::nn::{model::Model, perceptron::PerceptronModel};

fn main () {
    //Raw data: lr - 0.012, epochs - 70
    //Normal data: lr - 1.1 epochs - 15
    let mut model = PerceptronModel::new(1, 1.0);
    let (inputs, data) = ([vec![-20.0], vec![-10.0], vec![0.0], vec![10.0], vec![20.0]], [-4.0, 14.0, 32.0, 50.0 ,68.0]);
    println!("{:?}", inputs);
    model.fit(&inputs, &data, 15);
    let result = model.evaluate(&inputs, &data);
    println!("{:?}", result);
    println!("{:?}", model.predict(&vec![30.0]));
}