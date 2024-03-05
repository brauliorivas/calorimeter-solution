# Dependencies
using Flux
using CSV
using Random
using Statistics
using DataFrames

# Load data from csv and encode labels
function load_data(filename)
    # Load data from CSV file
    data = CSV.File(filename, header=true) |> DataFrame

    # Extract features (x, y, z) and labels
    features = [data[i, j] for i in 1:size(data, 1), j in 1:3]
    features = Float32.(features)  
    labels = map(label -> label == "s" ? 1 : 0, data[:, 4])  # Encode labels as integers (1 for "s", 0 for "b")

    return features, labels
end

# Normalize features for better training by subtracting mean and dividing by standard deviation (common practice)
function normalize(features)
    μ = mean(features, dims=2)
    σ = std(features, dims=2)
    return (features .- μ) ./ σ
end

# Split data into training and test sets using a given ratio
function split_data(features, labels, train_ratio)
    num_samples = size(features, 2)
    indices = shuffle(1:num_samples)
    train_size = Int(floor(train_ratio * num_samples))
    train_indices = indices[1:train_size]
    test_indices = indices[train_size+1:end]
    train_features = features[:, train_indices]
    train_labels = labels[train_indices]
    test_features = features[:, test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels
end

features, labels = load_data("dataset.csv") # Load data
features = normalize(features) # Normalize features
features = permutedims(features, [2, 1]) # Adjust dimensions for features
train_features, train_labels, test_features, test_labels = split_data(features, labels, 0.7) # Split data using a 70-30 ratio
train_labels = reshape(train_labels, 1, length(train_labels)) # Adjust dimensions for labels


# Define the neural network model
# after some testing, the simpler the model, the better the accuracy
# 4 layers with 3 neurons each and ReLU activation function performed the best
model = Chain(
    Dense(3, 3, relu), 
    Dense(3, 3, relu), 
    Dense(3, 3, relu), 
    Dense(3, 1),  # Output layer, 1 neuron for binary classification
    σ  # Sigmoid activation for binary classification
)

# Define loss function (binary cross-entropy loss)
loss(x, y) = Flux.binarycrossentropy(model(x), y)

# Define optimizer (ADAM)
optimizer = Flux.Optimise.ADAM()

# Training loop
epochs = 100 
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), [(train_features, train_labels)], optimizer)
end

# Evaluate the model on test data
accuracy = mean(Flux.onecold(model(test_features)) .== test_labels)
println("Test Accuracy: $accuracy")
