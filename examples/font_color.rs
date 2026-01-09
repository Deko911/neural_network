use neural_network::core::tensor::Tensor;
use neural_network::data::{self, reader};
use neural_network::nn::model::{Metrics, Model};
use neural_network::nn::perceptron::PerceptronModel;


fn main() {
    let data = data::reader::read_csv("data.csv");
    let (inputs, targets) = (reader::csv_to_matrix(&data, 3), reader::get_column(&data, 3));
    let inputs = Tensor::from_matrix(&inputs);
    let targets = Tensor::from_shape_vec((targets.len(), 1), targets);
    let mut model = PerceptronModel::new(3, 0.2, None);
    model.fit(&inputs, &targets, 10);
    let result = model.evaluate(&inputs, &targets);
    println!("{:?}", result);
    println!("{}", model.predict(&Tensor::from_vec(vec![17.0, 17.0, 17.0])));
}


#[allow(dead_code)]
fn alternative () {
    let mut data = data::reader::read_json("data.json");
    let [inputs, targets] = [data["inputs"].take(), data["targets"].take()];
    let targets: Vec<f32> = reader::json_to_array(&targets).unwrap();
    let inputs = reader::json_to_matrix(&inputs).unwrap();
    let inputs = Tensor::from_matrix(&inputs);
    let targets = Tensor::from_shape_vec((targets.len(), 1), targets);
    let mut model = PerceptronModel::new(3, 0.2, None);
    model.fit(&inputs, &targets, 10);
    let result = model.evaluate(&inputs, &targets);
    println!("{:?}", result);
    println!("{}", model.predict(&Tensor::from_vec(vec![17.0, 17.0, 17.0])));
}