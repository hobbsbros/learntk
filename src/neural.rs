//! Implements necessary neural network behaviors for LearnTK.

use std::fs;

#[allow(unused_imports)]
use crate::linalg::{
    Vector,
    Matrix,
    Activation,
    ReLU,
    Sigmoid,
    Softmax,
};

/// Holds a neural network layer.
pub struct Layer<const N: usize, const M: usize> {
    weights: Matrix<M, N>,
    biases: Vector<M>,
    activation: Box<dyn Activation<M>>,
    learning_rate: f64,
    input: Vector<N>,
    output: Vector<M>,
}

impl<const N: usize, const M: usize> Layer<N, M> {
    /// Constructs a new neural network layer.
    pub fn new(activation: Box<dyn Activation<M>>, learning_rate: f64) -> Self {
        Self {
            weights: Matrix::<M, N>::random(),
            biases: Vector::<M>::zero(),
            activation,
            learning_rate,
            input: Vector::<N>::zero(),
            output: Vector::<M>::zero(),
        }
    }

    /// Evaluate a vector input.
    pub fn evaluate(&mut self, vector: Vector<N>) -> Vector<M> {
        self.input = vector;
        self.output = self.activation.evaluate(self.weights.mult(vector) + self.biases);
        self.output
    }

    /// Adjusts this layer's weights and biases based on an error signal.
    pub fn backpropagate(&mut self, error: Vector<M>) -> Vector<N> {
        // This is the error of the raw output of this layer
        let raw_error = self.activation.backpropagate(error);

        // This is the change that should be made to the weight matrix
        let weight_adjust = raw_error.transpose_mult(self.input).scaled(self.learning_rate);

        // No bias adjustment yet (to be implemented)

        // This is the error signal to be propagated to the next layer
        let next_error = self.weights.transposed().mult(error);

        // Should this be added or subtracted?
        self.weights = self.weights - weight_adjust;

        next_error
    }
}

#[test]
fn evaluate_layer() {
    let mut layer = Layer::<3, 1>::new(Box::new(Sigmoid::<1>), 1.0);
    let vector = Vector::new([1.0, 1.0, 0.0]);
    let output = layer.evaluate(vector);
    let expected = Vector::new([1.0]);
    let error = output - expected;
    layer.backpropagate(error);

    dbg!(error);
    dbg!(output);

    let output = layer.evaluate(vector);
    let error = output - expected;

    dbg!(error);
    dbg!(output);
}

#[derive(Clone, Debug)]
/// Holds a training dataset.
pub struct TrainingDataset<const X: usize, const Y: usize> {
    data: Vec<(Vector<X>, Vector<Y>)>,
}

impl<const X: usize, const Y: usize> TrainingDataset<X, Y> {
    /// Constructs a new training dataset.
    pub fn new(data: Vec<(Vector<X>, Vector<Y>)>) -> Self {
        Self {
            data,
        }
    }

    /// Imports a training dataset from a file.
    pub fn import<'a>(filename: &'a str) -> Self {
        let raw = match fs::read_to_string(filename) {
            Ok(s) => s,
            Err(_) => String::new(),
        };

        let mut data = Vec::new();

        for line in raw.split("\n") {
            let cells = line.split("|")
                .map(|s| s.trim())
                .collect::<Vec<&str>>();
            let inputs = cells[0].split(" ")
                .map(|d| str::parse::<f64>(d).unwrap_or(0.0))
                .collect::<Vec<f64>>();
            let outputs = cells[1].split(" ")
                .map(|d| str::parse::<f64>(d).unwrap_or(0.0))
                .collect::<Vec<f64>>();

            let input = Vector::<X>::from(inputs);
            let expected = Vector::<Y>::from(outputs);

            data.push((input, expected));
        }

        Self {
            data,
        }
    }

    /// Yields the raw data from this dataset.
    pub fn yield_data(&self) -> Vec<(Vector<X>, Vector<Y>)> {
        self.data.to_owned()
    }
}

#[test]
fn import_dataset() {
    let dataset = TrainingDataset::<3, 1>::import("data.nntd");
    dbg!(dataset);
}

/// Holds a neural network.
pub struct Network<const X: usize, const H: usize, const Y: usize> {
    input_layer: Layer<X, H>,
    hidden_layers: Vec<Layer<H, H>>,
    output_layer: Layer<H, Y>,
    learning_rate: f64,
}

impl<const X: usize, const H: usize, const Y: usize> Network<X, H, Y> {
    /// Constructs a new neural network.
    pub fn new(
        input_activation: Box<dyn Activation<H>>,
        output_activation: Box<dyn Activation<Y>>,
        learning_rate: f64,
    ) -> Self {
        Self {
            input_layer: Layer::new(input_activation, learning_rate),
            hidden_layers: Vec::new(),
            output_layer: Layer::new(output_activation, learning_rate),
            learning_rate,
        }
    }

    /// Add a hidden layer.
    pub fn add_hidden_layer(&mut self, activation: Box<dyn Activation<H>>) {
        let layer = Layer::new(activation, self.learning_rate);
        self.hidden_layers.push(layer);
    }

    /// Evaluates a vector input.
    pub fn evaluate(&mut self, vector: Vector<X>) -> Vector<Y> {
        let mut output: Vector<H> = self.input_layer.evaluate(vector);

        for layer in self.hidden_layers.iter_mut() {
            output = layer.evaluate(output);
        }

        self.output_layer.evaluate(output)
    }

    /// Adjust this network's weights and biases based on an error signal.
    fn backpropagate(&mut self, error: Vector<Y>) {
        let mut error_signal: Vector<H> = self.output_layer.backpropagate(error);

        for layer in self.hidden_layers.iter_mut().rev() {
            error_signal = layer.backpropagate(error_signal);
        }

        self.input_layer.backpropagate(error_signal);
    }

    /// Train this network based on an input and an expected output.
    pub fn train(&mut self, input: Vector<X>, expected: Vector<Y>) -> f64 {
        let output = self.evaluate(input);
        let error = output - expected;
        self.backpropagate(error);
        0.5*error.norm()*error.norm()
    }

    /// Train this network based on a list of inputs and expected outputs.
    pub fn train_all(&mut self, dataset: TrainingDataset<X, Y>) -> f64 {
        let data = dataset.yield_data();

        let mut cost: f64 = 0.0;

        for &(input, expected) in &data {
            cost += self.train(input, expected);
        }
        
        cost/(data.len() as f64)
    }
}