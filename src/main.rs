use neural_network::core::tensor::Tensor;

fn main() {
    let a = Tensor::from_elem(1.0);
    println!("{:?}", a.shape());
    let b = Tensor::from_vec(&vec![1.0, 2.0, 3.0, 4.0]);
    println!("{:?}", b.shape());
    let mut c = Tensor::from_matrix(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    println!("{:?}", c.shape());
    let d = Tensor::zeros((3, 2));
    println!("{}", d);
    println!("{}", a.clone() + b.clone());
    println!("{}", b.clone() - a.clone());
    println!("{}", a.clone() * b.clone());
    println!("{}", a.clone() / b.clone());
    c.iter_mut().for_each(|x| *x = *x * 2.0);
    println!("{}", c);
}
