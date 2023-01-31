//! Main library for the LearnTK machine learning toolkit.

use std::ops::{
    Index,
    IndexMut,
};

#[derive(Clone, Copy, PartialEq, Debug)]
/// Stores a column vector of floating-point values.
pub struct Vector<const N: usize> ([f64; N]);

/// Indexes into a vector.
impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

/// Indexes mutably into a vector.
impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0[i]
    }
}

impl <const N: usize> Vector<N> {
    /// Constructs a new vector.
    pub fn new(values: [f64; N]) -> Self {
        Self (values)
    }

    /// Constructs a new zero vector.
    pub fn zero() -> Self {
        Self ([0.0; N])
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
/// Stores a matrix of floating-point values.
pub struct Matrix<const N: usize, const M: usize> ([[f64; M]; N]);

/// Indexes into a matrix.
impl<const N: usize, const M: usize> Index<(usize, usize)> for Matrix<N, M> {
    type Output = f64;

    fn index(&self, i: (usize, usize)) -> &Self::Output {
        &self.0[i.0][i.1]
    }
}

/// Indexes mutably into a matrix.
impl<const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<N, M> {
    fn index_mut(&mut self, i: (usize, usize)) -> &mut Self::Output {
        &mut self.0[i.0][i.1]
    }
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    /// Constructs a new matrix.
    pub fn new(values: [[f64; M]; N]) -> Self {
        Self (values)
    }

    /// Constructs a new zero matrix.
    pub fn zero() -> Self {
        Self ([[0.0; M]; N])
    }

    /// Right-multiplies this matrix by a vector, returning a vector.
    pub fn mult(&self, vector: Vector<M>) -> Vector<N> {
        // Construct the output vector
        let mut output = Vector::<N>::zero();

        for i in 0..N {
            for j in 0..M {
                output[i] += self[(i, j)]*vector[j];
            }
        }

        output
    }

    /// Returns a transposed copy of this matrix.
    pub fn transposed(&self) -> Matrix<M, N> {
        let mut output = Matrix::<M, N>::zero();

        for i in 0..N {
            for j in 0..M {
                output[(j, i)] = self[(i, j)];
            }
        }

        output
    }
}

#[test]
fn multiply_matrix_and_vector() {
    let vector = Vector::<3>::new([-1.0, 2.0, 0.0]);
    let matrix = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let output = Vector::<2>::new([3.0, 6.0]);
    assert_eq!(matrix.mult(vector), output);
}

/// Holds a list of activation functions for vectors.
pub struct Activation<const N: usize>;

impl<const N: usize> Activation<N> {
    #![allow(unused_variables)]
    
    /// Performs the sigmoid (logistic) activation function on the vector given.
    pub fn sigmoid(vector: Vector<N>) -> Vector<N> {
        todo!()
    }

    /// Performs the ReLU activation function on the vector given.
    pub fn relu(vector: Vector<N>) -> Vector<N> {
        todo!()
    }

    /// Performs the softmax activation function on the vector given.
    pub fn softmax(vector: Vector<N>) -> Vector<N> {
        todo!()
    }
}