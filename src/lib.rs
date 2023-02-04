//! Main library for the LearnTK package.

mod linalg;

use linalg::{
    Vector,
    Matrix,
    Activation,
    ReLU,
};

/// Holds a neural network layer.
pub struct Layer<const N: usize, const M: usize> {
    weights: Matrix<M, N>,
    biases: Vector<M>,
    activation: Box<dyn Activation<M>>,
    learning_rate: f64,
}

impl<const N: usize, const M: usize> Layer<N, M> {
    /// Constructs a new neural network layer.
    pub fn new(activation: Box<dyn Activation<M>>, learning_rate: f64) -> Self {
        Self {
            weights: Matrix::<M, N>::random(),
            biases: Vector::<M>::random(),
            activation,
            learning_rate,
        }
    }

    /// Evaluate a vector input.
    pub fn evaluate(&self, vector: Vector<N>) -> Vector<M> {
        self.activation.evaluate(self.weights.mult(vector) + self.biases)
    }

    /// Adjusts this layer's weights and biases based on an error signal.
    pub fn backpropagate(&mut self, error: Vector<M>) -> Vector<N> {
        self.weights.transposed().mult(error)
    }
}

#[test]
fn evaluate_layer() {
    let layer = Layer::<6, 6>::new(Box::new(ReLU::<6>), 1.0);
    let vector = Vector::new([0.3, 0.6, 0.1, 0.3, 0.6, 1.0]);
    dbg!(&layer.evaluate(vector));
}