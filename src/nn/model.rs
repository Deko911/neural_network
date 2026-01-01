pub trait Model {
    type Input;
    type Output;

    fn predict(&self, input: &Self::Input) -> Self::Output;
    fn evaluate(&self, input: &[Self::Input], target: &[Self::Output]) -> f32;
    fn evaluate_one(&self, input: &Self::Input, target: &Self::Output) -> f32;
    fn fit(&mut self, input: &[Self::Input], target: &[Self::Output], epochs: usize);
    fn fit_raw(&mut self, input: &[Self::Input], target: &[Self::Output], epochs: usize);
}

pub trait Trainable {
    type Input;
    type Output;

    fn predict(&self, input: &Self::Input) -> Self::Output;
    fn error(&self, input: &Self::Input, target: &Self::Output) -> f32;
    fn train_step(&mut self, input: &[Self::Input], target: &[Self::Output]);
}
