use neural_network::core::tensor::Tensor;
use neural_network::nn::logistic_regression::LogisticRegression;
use neural_network::nn::model::Model;


fn main() {
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
    let mut model = LogisticRegression::new(3, 0.2, None);
    model.fit(&inputs, &targets, 10);
    let result = model.evaluate(&inputs, &targets);
    println!("{:?}", result);
    let student = Tensor::from_vec(vec![5.0, 75.0, 68.0]);
    println!("{:?}", model.predict_prob(&student));
    println!("{}", model.choice(&student, &["disapprove", "approve"]));
}
