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

    let nn = NeuralNetworkModel::load_model("models/numbers.json");
    if let Err(_) = nn {
        new_nn(dataset);
        return;
    }

    let nn = nn.unwrap();

    let test_inputs = &dataset.test_img;
    let test_targets = &dataset.test_label;

    //Cost 0.26 - lr 0.17 - 30 epochs - batch_size 100
    println!("{}", nn.evaluate(&test_inputs, &test_targets));

    //Evaluate one number

    let image_num = 1;
    let number = dataset.test_img.row(image_num);

    println!("{}", dataset.get_number_matrix(image_num, false));
    println!("{}", test_targets.row(image_num));
    println!("{}", nn.predict(&number));

    // Accurate for all tests 94.28% - lr 0.17 - 30 epochs - batch_size 100
    let size = test_inputs.len();
    let mut accurate = 0.0;

    for i in 0..size {
        let input = test_inputs.row(i);
        let target_vec = test_targets.row(i).to_vec();
        let mut target = 0;
        for (i, t) in target_vec.iter().enumerate() {
            if *t == 1.0 {
                target = i; 
            }
        }
        
        let predict_vec = nn.predict(&input).to_vec();
        let mut max_predict = predict_vec[0];
        let mut predict = 0;
        for (i, p) in predict_vec.iter().enumerate() {
            if *p > max_predict {
                max_predict = *p;
                predict = i;
            }
        }
        accurate += if predict == target { 1.0 } else { 0.0 }
    }
    accurate = accurate / size as f32;
    println!("Accurate in {} tests: {}%", size, accurate * 100.0);

}

fn new_nn(dataset: ImageDataset) {

    let image_num = 5;
    let number = dataset.test_img.row(image_num);

    let training_inputs = &dataset.training_img;
    let training_targets = &dataset.training_label;
    let mut nn = NeuralNetworkModel::new(vec![
        Layer::new((784, 16), ACTIVATIONS::RELU, INITIALIZER::HE),
        Layer::new((16, 16), ACTIVATIONS::RELU, INITIALIZER::HE),
        Layer::new((16, 10), ACTIVATIONS::SOFTMAX, INITIALIZER::XAVIER)
        ], 0.17, Some(neural_network::nn::loss::LOSS::CROSS_ENTROPY));
    nn.fit(training_inputs, training_targets, 30, 100);

    let test_inputs = &dataset.test_img;
    let test_targets = &dataset.test_label;

    println!("{}", nn.evaluate(&test_inputs, &test_targets));
    println!("{}", dataset.get_number_matrix(image_num, false));
    println!("{}", test_targets.row(image_num));
    println!("{}", nn.predict(&number));
    nn.save("models/numbers.json");
}

