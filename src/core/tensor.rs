use std::ops::*;
use std::fmt::{Display, Debug};

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
    
    pub fn from_vec(vec: &Vec<f32>) -> Self {
        let data = ndarray::arr1(vec).into_dyn();
        Self { data }
    }

    pub fn from_matrix(mat: &Vec<Vec<f32>>) -> Self{
        let rows = mat.len();
        let cols = mat.get(0).unwrap_or(&vec![]).len();
        let flat: Vec<f32> = mat.iter().flat_map(|row| row.iter().copied()).collect();
        let data = ndarray::Array2::from_shape_vec((rows, cols), flat).unwrap().into_dyn();
        Self { data }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let data = ndarray::Array::zeros(shape).into_dyn();
        Self { data }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
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

    pub fn map<F: FnMut(&f32) -> f32>(&mut self, f: F) -> Self {
        let data = self.data.map(f).into_dyn();
        Self { data }
    }

}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(data: {}, shape: {:?})", self.data, self.shape())
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)    
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data + rhs.data
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

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data * rhs.data 
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
