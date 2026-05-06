use neural_network::{core::tensor::Tensor, data::scaler::StandardScaler};

fn main() {
    let data = Tensor::from_matrix(&vec![
        vec![1.0, 10.0, 100.0],
        vec![2.0, 20.0, 200.0],
        vec![3.0, 30.0, 300.0]
    ]);
    let mut std = StandardScaler::new();
    std.fit(&data);
    let transform = std.transform(&data);
    println!("transform {}", transform);
    println!("untransform {}", std.inverse_transform(&transform.row(2)));
}