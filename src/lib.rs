//! Main library for the LearnTK machine learning package.

mod linalg;
mod neural;

pub use linalg::{
    Sigmoid,
    ReLU,
    Softmax,
};

pub use neural::{
    TrainingDataset,
    Network,
};