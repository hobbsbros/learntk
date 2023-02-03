//! Linear algebra package for LearnTK.

use std::ops::{
    Index,
    IndexMut,
};

use rand::random;

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

impl<const N: usize> Vector<N> {
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

    /// Constructs a new, random matrix.
    pub fn random() -> Self {
        let mut output = [[0.0f64; M]; N];

        for i in 0..N {
            for j in 0..M {
                output[j][i] = 2.0*random::<f64>() - 1.0;
            }
        }

        Self (output)
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

pub trait Activation<const N: usize> {
    /// Performs the activation function on the vector given.
    fn evaluate(&self, vector: Vector<N>) -> Vector<N>;

    /// Performs the backpropagation function on the vector given.
    fn backpropagate(&self, vector: Vector<N>) -> Vector<N>;
}

/// Holds a sigmoid activation function for vectors.
pub struct Sigmoid<const N: usize>;

impl<const N: usize> Activation<N> for Sigmoid<N> {
    /// Performs the sigmoid (logistic) activation function on the vector given.
    fn evaluate(&self, vector: Vector<N>) -> Vector<N> {
        let mut output = Vector::<N>::zero();

        for i in 0..N {
            output[i] = 1.0/(1.0 + (-vector[i]).exp());
        }

        output
    }

    /// Performs the sigmoid (logistic) backpropagation function on the vector given.
    fn backpropagate(&self, vector: Vector<N>) -> Vector<N> {
        todo!()
    }
}

/// Holds a ReLU activation function for vectors.
pub struct ReLU<const N: usize>;

impl<const N: usize> Activation<N> for ReLU<N> {
    /// Performs the ReLU activation function on the vector given.
    fn evaluate(&self, vector: Vector<N>) -> Vector<N> {
        let mut output = Vector::<N>::zero();

        for i in 0..N {
            if vector[i] >= 0.0 {
                output[i] = vector[i];
            }
        }

        output
    }

    /// Performs the ReLU backpropagation function on the vector given.
    fn backpropagate(&self, vector: Vector<N>) -> Vector<N> {
        todo!()
    }
}

/// Holds a softmax activation function for vectors.
pub struct Softmax<const N: usize>;

impl<const N: usize> Activation<N> for Softmax<N> {
    /// Performs the softmax activation function on the vector given.
    fn evaluate(&self, vector: Vector<N>) -> Vector<N> {
        let mut output = Vector::<N>::zero();
        let mut total: f64 = 0.0;

        for i in 0..N {
            total += vector[i].exp();
        }

        for i in 0..N {
            output[i] = vector[i].exp()/total;
        }

        output
    }

    /// Performs the softmax backpropagation function on the vector given.
    fn backpropagate(&self, vector: Vector<N>) -> Vector<N> {
        todo!()
    }
}