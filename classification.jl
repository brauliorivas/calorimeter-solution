using Flux
using CSV
using Random
using Statistics
using DataFrames

# Load and preprocess the data
function load_data(filename)
    # Load data from CSV file
    data = CSV.File(filename, header=true) |> DataFrame

    # Extract features (x, y, z) and labels
    features = [data[i, j] for i in 1:size(data, 1), j in 1:3]
    features = Float32.(features)  
    labels = map(label -> label == "s" ? 1 : 0, data[:, 4])  # Encode labels as integers (1 for "s", 0 for "b")

    return features, labels
end

function normalize(features)
    μ = mean(features, dims=2)
    σ = std(features, dims=2)
    return (features .- μ) ./ σ
end

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

features, labels = load_data("dataset.csv")
features = normalize(features)
features = permutedims(features, [2, 1])
train_features, train_labels, test_features, test_labels = split_data(features, labels, 0.7)
train_labels = reshape(train_labels, 1, length(train_labels))


# Define the neural network model
model = Chain(
    Dense(3, 64, relu),  # Input size: 3, Output size: 64
    Dense(64, 32, relu),  # Hidden layer
    Dense(32, 1),  # Output layer, 1 neuron for binary classification
    σ  # Sigmoid activation for binary classification
)

# Define loss function (binary cross-entropy loss)
loss(x, y) = Flux.binarycrossentropy(model(x), y)

# Define optimizer (e.g., Adam)
optimizer = Flux.Optimise.ADAM()

# Training loop
epochs = 100
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), [(train_features, train_labels)], optimizer)
end

# Evaluate the model on test data
accuracy = mean(Flux.onecold(model(test_features)) .== test_labels)
println("Test Accuracy: $accuracy")
