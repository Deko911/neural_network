use mnist::*;
use neural_network::core::tensor::Tensor;
use neural_network::nn::layer::Layer;
use neural_network::nn::model::{Metrics, Model};
use neural_network::nn::neural_network::NeuralNetworkModel;
use neural_network::nn::activation::ACTIVATIONS;
use neural_network::nn::layer::INITIALIZER;

const BASE_URL: &'static str = "data_sets/";

struct ImageDataset {
    pub training_img: Tensor,
    pub training_label: Tensor,
    pub test_img: Tensor,
    pub test_label: Tensor,
}

impl ImageDataset {
    pub fn new(training_length: u32, test_length: u32) -> Self {
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(training_length)
            .validation_set_length(10_000)
            .test_set_length(test_length)
            .base_path(BASE_URL)
            .finalize();

        let training_img = Tensor::from_shape_vec((training_length as usize, 784), trn_img.iter().map(|x| *x as f32 / 256.0).collect());
        let training_label = Tensor::from_matrix(&trn_lbl.iter().map(|x| {
            let mut vec = [0.0 as f32; 10].to_vec();
            vec[*x as usize] = 1.0;
            vec
        }).collect());
        let test_img = Tensor::from_shape_vec((test_length as usize, 784), tst_img.iter().map(|x| *x as f32 / 256.0).collect());
        let test_label = Tensor::from_matrix(&tst_lbl.iter().map(|x| {
            let mut vec = [0.0 as f32; 10].to_vec();
            vec[*x as usize] = 1.0;
            vec
        }).collect());

        Self { training_img, training_label, test_img, test_label }
    }

    pub fn get_number_matrix(&self, number: usize, training: bool) -> Tensor {
        let tensor = if training { &self.training_img } else { &self.test_img };
        let vec = tensor.row(number).to_vec();
        Tensor::from_shape_vec((28, 28), vec)
    }
}

fn main() {
    let dataset = ImageDataset::new(50_000, 10_000);

    let image_num = 5;
    let number = dataset.test_img.row(image_num);

    let training_inputs = &dataset.training_img;
    let training_targets = &dataset.training_label;
    let mut nn = NeuralNetworkModel::new(vec![
        Layer::new((784, 16), ACTIVATIONS::SIGMOID, INITIALIZER::XAVIER),
        Layer::new((16, 16), ACTIVATIONS::SIGMOID, INITIALIZER::XAVIER),
        Layer::new((16, 10), ACTIVATIONS::SOFTMAX, INITIALIZER::XAVIER)
        ], 0.5, Some(neural_network::nn::loss::LOSS::CROSS_ENTROPY));
    nn.fit(training_inputs, training_targets, 20, 100);

    let test_inputs = &dataset.test_img;
    let test_targets = &dataset.test_label;

    println!("{}", nn.evaluate(&test_inputs, &test_targets));
    println!("{}", dataset.get_number_matrix(image_num, false));
    println!("{}", test_targets.row(image_num));
    println!("{}", nn.predict(&number));
}
