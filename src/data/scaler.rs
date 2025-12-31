use std::{iter::Sum, ops::Div};

pub struct StandardScale {
    mean: f32,
    std: f32,
}

impl StandardScale {
    pub fn new<T>(data: &[Vec<T>]) -> Self
    where
        T: PartialOrd,
        T: Copy,
        T: Default,
        T: Div<Output = T>,
        T: Sum,
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
        println!("{}, {}", mean, std);
        Self { mean, std }
    }

    pub fn transform<T>(&self, x: T) -> f32
    where
        f32: From<T>,
    {
        (f32::from(x) - self.mean) / self.std
    }

    pub fn transform_batch<T>(&self, data: &[Vec<T>]) -> Vec<Vec<f32>>
    where
        T: PartialOrd,
        T: Copy,
        T: Default,
        T: Div<Output = T>,
        T: Sum,
        f32: From<T>,
    {
        data.iter().map(|r| r.iter().map(|x| self.transform(*x)).collect()).collect()
    }

    pub fn inverse_transform<T>(&self, x: f32) -> T
    where
        T: From<f32> 
    {
        T::from(x * self.std + self.mean)
    }
}
