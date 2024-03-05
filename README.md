# Neural Network for Binary Classification

This code aims to solve the task for GSoC given by [CERN](https://github.com/graeme-a-stewart/hsf-julia-ml-gsoc). The basic task is to train a machine learning model in Julia to classify a ternary of x, y, and z values into signal or background labels. 

### To run this code
1. Clone the repository:
   ```sh
   git clone git@github.com:brauliorivas/calorimeter-solution.git
   ```

2. Navigate to the cloned directory:
   ```sh
   cd calorimeter-solution
   ```

3. Unzip the dataset:
   ```sh
   unzip dataset.zip
   ```

4. Open Julia's REPL (Read-Eval-Print Loop):
   ```sh
   julia
   ```

5. Once in the Julia REPL, activate the project environment:
   ```julia
   ]
   ```

6. Once in the package mode, activate the project environment:
   ```julia
   activate .
   ```

7. Install the required packages:
   ```julia
   instantiate
   ```

8. Exit the package mode and return to the Julia REPL:
   ```julia
   ctrl + c
   ```

9. Run the script:
   ```sh
   include("classification.jl")
   ```

## Neural Network Architecture

The neural network architecture is designed to classify input data into one of two classes: "s" (signal) or "b" (background). It consists of three dense layers:

1. **Input Layer:** 
   - Number of neurons: 3 input - 3 output
   - Activation function: ReLU (Rectified Linear Unit) (common activation function used in various neural networks)
   - Purpose: This layer accepts input data with three features representing the x, y, and z dimensions of the input samples.

2. **Hidden Layers:** 
   The simpler, the better. After trying more neurons, the accuracy was not so good. But later, with lower neurons and more layers, it improved. 
   - First Hidden Layer:
     - Number of neurons: 3 input - 3 output
     - Activation function: ReLU 
   - Second Hidden Layer:
     - Number of neurons: 3 input - 3 output
     - Activation function: ReLU
   - Purpose: These layers perform nonlinear transformations on the input data, extracting relevant features for classification.

3. **Output Layer:**
   - Number of neurons: 3 input - 1 output
   - Activation function: Sigmoid
   - Purpose: This layer produces the final output, which represents the probability that the input sample belongs to class "s" (signal) or "b" (background). A sigmoid activation function is used to ensure the output is between 0 and 1, allowing for interpretation as a probability, where 1 equals "s" and 0 equals "b". 

## Training Procedure

The model is trained using an optimization algorithm known as ADAM. This algorithm is a variant of stochastic gradient descent (SGD) that maintains separate adaptive learning rates for each parameter. ADAM is well-suited for training deep neural networks and typically converges faster than traditional SGD.

During training, the model adjusts its parameters (weights and biases) iteratively to minimize a predefined loss function. This process involves passing training data through the network, computing the loss between the predicted outputs and the true labels, and updating the parameters using gradient descent.

## Choice of Loss Function

The binary cross-entropy loss function, also known as log loss, is employed for this task. This loss function is commonly used in binary classification problems, where the goal is to predict one of two mutually exclusive classes.

## Accuracy
The accuracy for this model is ~ 0.227

## Dependencies

- Flux: A Julia library for machine learning.
- CSV: A package for reading and writing CSV files.
- Random: Provides functions for generating random numbers.
- Statistics: Includes statistical functions like mean and standard deviation.
- DataFrames: Used for working with tabular data.

## License

This code is provided under the [MIT License](LICENSE).
