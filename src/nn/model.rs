pub trait Model {
    type Input;
    type Output;
    fn predict(&self, input: &Self::Input) -> Self::Output;
    fn evaluate(&self, input: &[Self::Input], target: &[Self::Output]) -> f32;
    fn evaluate_one(&self, input: &Self::Input, target: &Self::Output) -> f32;
}

pub trait Trainable: Model {
    fn train_step(&mut self, input: &[Self::Input], target: &[Self::Output]);

    fn fit(&mut self, input: &[Self::Input], target: &[Self::Output], epochs: usize) {
        for _ in 0..epochs {
            self.train_step(input, target);
        }
    }
}
