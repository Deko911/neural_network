pub struct StandardScale {
    mean: f32,
    std: f32,
}

impl StandardScale {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
        }
    }

    pub fn fit<T>(&mut self, data: &[Vec<T>])
    where
        T: Copy,
        f32: From<T>,
    {
        let mut count = 0;
        let mean = data
            .iter()
            .map(|r| {
                count += r.len();
                r.iter().map(|x| f32::from(*x)).sum::<f32>()
            })
            .sum::<f32>()
            / count as f32;
        let sum = data
            .iter()
            .map(|r| {
                r.iter()
                    .map(|x| (f32::from(*x) - mean).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / count as f32;
        let std = sum.sqrt().max(1e-8);
        self.mean = mean;
        self.std = std;
    }

    pub fn transform<T>(&self, input: &[T]) -> Vec<f32>
    where
        T: Copy,
        f32: From<T>,
    {
        input.iter().map(|x| (f32::from(*x) - self.mean) / self.std).collect()
    }

    pub fn transform_batch<T>(&self, data: &[Vec<T>]) -> Vec<Vec<f32>>
    where
        T: Copy,
        f32: From<T>,
    {
        data.iter()
            .map(|r| self.transform(&r))
            .collect()
    }

    pub fn inverse_transform<T>(&self, x: f32) -> T
    where
        T: From<f32>,
    {
        T::from(x * self.std + self.mean)
    }
}
