use neural_network::data::{self, reader};
use neural_network::nn::model::Model;
use neural_network::nn::perceptron::PerceptronModel;

fn main () {
    let data = data::reader::read_csv("data.csv");
    let (inputs, targets) = (reader::csv_to_matrix(&data, 3), reader::get_column(&data, 3));
    let mut model = PerceptronModel::new(3, 0.1);
    model.fit(&inputs, &targets, 50);
    let result = model.evaluate(&inputs, &targets);
    println!("{:?}", result);
    println!("{:?}", model.predict(&vec![17.0, 17.0, 17.0]));
    
}