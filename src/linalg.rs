//! Linear algebra package for LearnTK.

use std::{
    ops::{
        Index,
        IndexMut,
        Add,
        Sub,
    },
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

/// Implements addition of two vectors.
impl<const N: usize> Add for Vector<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut output = Self::zero();

        for i in 0..N {
            output[i] = self[i] + other[i];
        }

        output
    }
}

/// Implements subtraction of two vectors.
impl<const N: usize> Sub for Vector<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut output = Self::zero();

        for i in 0..N {
            output[i] = self[i] - other[i];
        }

        output
    }
}

impl<const N: usize> Vector<N> {
    /// Constructs a new vector.
    pub fn new(values: [f64; N]) -> Self {
        Self (values)
    }

    /// Constructs a new vector from a `Vec<f64>`.
    pub fn from(values: Vec<f64>) -> Self {
        let mut output = [0.0f64; N];

        for i in 0..values.len() {
            output[i] = values[i];
        }

        Self (output)
    }

    /// Constructs a new zero vector.
    pub fn zero() -> Self {
        Self ([0.0; N])
    }

    /// Constructs a new, random vector.
    pub fn random() -> Self {
        let mut output = [0.0f64; N];

        for i in 0..N {
            output[i] = 2.0*random::<f64>() - 1.0;
        }

        Self (output)
    }

    /// Right-multiplies this vector according to the Hadamard product.
    pub fn mult(&self, other: Self) -> Self {
        let mut output = Self::zero();

        for i in 0..N {
            output[i] = self[i]*other[i];
        }

        output
    }

    /// Right-multiplies this vector by the transpose of the input, constructing a matrix.
    pub fn transpose_mult<const M: usize>(&self, other: Vector<M>) -> Matrix<N, M> {
        let mut output = Matrix::<N, M>::zero();

        for i in 0..N {
            for j in 0..M {
                output[(i, j)] = self[i]*other[j];
            }
        }

        output
    }

    /// Computes the magnitude of this vector.
    pub fn norm(&self) -> f64 {
        let mut output: f64 = 0.0;

        for i in 0..N {
            output += self[i]*self[i];
        }

        (output/(N as f64)).max(0.0).sqrt()
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

/// Implements addition of two matrices.
impl<const N: usize, const M: usize> Add for Matrix<N, M> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut output = Self::zero();

        for i in 0..N {
            for j in 0..M {
                output[(i, j)] = self[(i, j)] + other[(i, j)];
            }
        }

        output
    }
}

/// Implements subtraction of two matrices.
impl<const N: usize, const M: usize> Sub for Matrix<N, M> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut output = Self::zero();

        for i in 0..N {
            for j in 0..M {
                output[(i, j)] = self[(i, j)] - other[(i, j)];
            }
        }

        output
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
        let mut output = Self::zero();

        for i in 0..N {
            for j in 0..M {
                output[(i, j)] = 2.0*random::<f64>() - 1.0;
            }
        }

        output
    }

    /// Scales this matrix by a scalar.
    pub fn scaled(&self, scalar: f64) -> Self {
        let mut output = Self::zero();

        for i in 0..N {
            for j in 0..M {
                output[(i, j)] = scalar*self[(i, j)];
            }
        }

        output
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
            output[i] = 1.0/(1.0 + (-vector[i].clamp(-100.0, 100.0)).exp());
        }

        output
    }

    /// Performs the sigmoid (logistic) backpropagation function on the vector given.
    fn backpropagate(&self, vector: Vector<N>) -> Vector<N> {
        let mut output = Vector::<N>::zero();

        for i in 0..N {
            output[i] = vector[i]*(1.0 - vector[i]);
        }

        output
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
        let mut output = Vector::<N>::zero();

        for i in 0..N {
            if vector[i] >= 0.0 {
                output[i] = 1.0;
            } else {
                output[i] = 0.0;
            }
        }

        output
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

    #[allow(unused_variables)]
    /// Performs the softmax backpropagation function on the vector given.
    fn backpropagate(&self, vector: Vector<N>) -> Vector<N> {
        todo!()
    }
}