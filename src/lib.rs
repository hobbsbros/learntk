//! Main library for the LearnTK package.

mod linalg;

use linalg::{
    Vector,
    Matrix,
    Activation,
};

/// Holds a neural network layer.
pub struct Layer<const N: usize, const M: usize> {
    weights: Matrix<M, N>,
    activation: Box<dyn Activation<M>>,
}

impl<const N: usize, const M: usize> Layer<N, M> {
    /// Constructs a new neural network layer.
    pub fn new(&self, activation: Box<dyn Activation<M>>) -> Self {
        todo!()
    }
}