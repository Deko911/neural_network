use neural_network::{core::tensor::Tensor, nn::perceptron::PerceptronModel};
use neural_network::nn::model::{Metrics, Model};


fn main () {
    //Raw data: lr - 0.012, epochs - 70
    //Normal data: lr - 1.0 epochs - 15
    let mut model = PerceptronModel::new(1, 1.0, None);
    let inputs = Tensor::from_vec_col(vec![-20.0, -10.0, 0.0, 10.0, 20.0]);
    let data = Tensor::from_vec_col(vec![-4.0, 14.0, 32.0, 50.0, 68.0]);
    println!("{}", inputs);
    model.fit(&inputs, &data, 15);
    let result = model.evaluate(&inputs, &data);
    println!("{:?}", result);
    println!("{}", model.predict(&Tensor::from_elem(30.0)));
}