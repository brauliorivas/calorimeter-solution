# Neural Network for Binary Classification

## Neural Network Architecture

The neural network architecture is designed to classify input data into one of two classes: "s" (signal) or "b" (background). It consists of three dense layers:

1. **Input Layer:** 
   - Number of neurons: 3
   - Purpose: This layer accepts input data with three features representing the x, y, and z dimensions of the input samples.

2. **Hidden Layers:** 
   - First Hidden Layer:
     - Number of neurons: 64
     - Activation function: ReLU (Rectified Linear Unit)
   - Second Hidden Layer:
     - Number of neurons: 32
     - Activation function: ReLU
   - Purpose: These layers perform nonlinear transformations on the input data, extracting relevant features for classification.

3. **Output Layer:**
   - Number of neurons: 1
   - Activation function: Sigmoid
   - Purpose: This layer produces the final output, which represents the probability that the input sample belongs to class "s" (signal) or "b" (background). A sigmoid activation function is used to ensure the output is between 0 and 1, allowing for interpretation as a probability, where 1 equals "s" and 0 equals "b". 

## Training Procedure

The model is trained using an optimization algorithm known as ADAM. This algorithm is a variant of stochastic gradient descent (SGD) that maintains separate adaptive learning rates for each parameter. ADAM is well-suited for training deep neural networks and typically converges faster than traditional SGD.

During training, the model adjusts its parameters (weights and biases) iteratively to minimize a predefined loss function. This process involves passing training data through the network, computing the loss between the predicted outputs and the true labels, and updating the parameters using gradient descent.

## Choice of Loss Function

The binary cross-entropy loss function, also known as log loss, is employed for this task. This loss function is commonly used in binary classification problems, where the goal is to predict one of two mutually exclusive classes.

## Dependencies

- Flux: A Julia library for machine learning.
- CSV: A package for reading and writing CSV files.
- Random: Provides functions for generating random numbers.
- Statistics: Includes statistical functions like mean and standard deviation.
- DataFrames: Used for working with tabular data.

## License

This code is provided under the [MIT License](LICENSE).
