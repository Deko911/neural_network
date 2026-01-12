use std::ops::*;
use std::fmt::{Display, Debug};

use ndarray::linalg::Dot;
use ndarray::{Dim, IxDynImpl};
use ndarray::iter::{Iter, IterMut};

#[derive(Clone)]
pub struct Tensor {
    pub data: ndarray::ArrayD<f32>
}

impl Tensor {
    pub fn from_elem(elem: f32) -> Self {
        let data = ndarray::arr0(elem).into_dyn();
        Self { data }
    }
    
    pub fn from_vec(vec: Vec<f32>) -> Self {
        Tensor::from_shape_vec((1, vec.len()), vec)
    }

    pub fn from_vec_col(vec: Vec<f32>) -> Self {
        Tensor::from_shape_vec((vec.len(), 1), vec)
    }

    pub fn from_matrix(mat: &Vec<Vec<f32>>) -> Self{
        let rows = mat.len();
        let cols = mat.get(0).unwrap_or(&vec![]).len();
        let flat: Vec<f32> = mat.iter().flat_map(|row| row.iter().copied()).collect();
        let data = ndarray::Array2::from_shape_vec((rows, cols), flat).unwrap().into_dyn();
        Self { data }
    }

    pub fn from_shape_vec (shape: (usize, usize), vec: Vec<f32>) -> Self {
        let data = ndarray::Array2::from_shape_vec(shape, vec).unwrap().into_dyn();
        Self { data }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let data = ndarray::Array::zeros(shape).into_dyn();
        Self { data }
    }

    pub fn random(shape: (usize, usize)) -> Self {
        let mut vec = vec![0.0f32; shape.0 * shape.1];
        rand::fill(&mut vec[..]);
        let t = Self::from_shape_vec(shape, vec);
        t * 2.0 - 1.0 
    }

    pub fn as_slice(&self) -> Option<&[f32]> {
        self.data.as_slice()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.iter().map(|x| *x).collect()
    }

    pub fn as_f32(&self) -> f32 {
        self.data.first().unwrap().clone()
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn shape_tuple(&self) -> (usize, usize) {
        let shape = self.shape();
        let rows = *shape.get(0).unwrap_or(&1);
        let cols = *shape.get(1).unwrap_or(&1);
        (rows, cols)
    }

    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn row(&self, idx: usize) -> Tensor {
        let data = self.data.slice(ndarray::s![idx..idx+1, ..]).into_dyn().to_owned();
        Self { data }
    }

    pub fn dot(&self, rhs: &Self) -> Self {
        let data = self.data.dot(&rhs.data).into_dyn();
        Self { data }
    }

    pub fn t(&self) -> Self {
        let data = self.data.t().to_owned().into_dyn();
        Self { data }
    }

    pub fn iter(&self) -> Iter<'_, f32, Dim<IxDynImpl>> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, f32, Dim<IxDynImpl>> {
        self.data.iter_mut()
    }

    pub fn for_each<F: FnMut(&f32) -> ()>(&mut self, f: F) {
        self.data.for_each(f);
    }

    pub fn map<F: FnMut(&f32) -> f32>(&self, f: F) -> Self {
        let data = self.data.map(f).into_dyn();
        Self { data }
    }

}

//Display
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(data: {}, shape: {:?})", self.data, self.shape())
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.size() == 1 {
            write!(f, "{}", self.as_f32()) 
        } else {
            write!(f, "{}", self.data) 
        }
    }
}

//Math
impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data + rhs.data
        }
    }
}

impl Add<f32> for Tensor {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        Self {
            data: self.data + rhs
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data + &rhs.data
        }
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f32) -> Self::Output {
        Tensor {
            data: &self.data + rhs
        }
    }
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data - rhs.data
        }
    }
}

impl Sub<f32> for Tensor {
    type Output = Self;
    fn sub(self, rhs: f32) -> Self::Output {
        Self {
            data: self.data - rhs
        }
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data - &rhs.data
        }
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f32) -> Self::Output {
        Tensor {
            data: &self.data - rhs
        }
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data * rhs.data 
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            data: self.data * rhs
        }
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data * &rhs.data 
        }
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Self::Output {
        Tensor {
            data: &self.data * rhs
        }
    }
}

impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data / rhs.data
        }
    }
}

impl Div<f32> for Tensor {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self {
            data: self.data / rhs
        }
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data / &rhs.data
        }
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f32) -> Self::Output {
        Tensor {
            data: &self.data / rhs
        }
    }
}

impl AddAssign for Tensor {
    fn add_assign(&mut self, rhs: Self) {
        self.data = &self.data + rhs.data
    }
}