#pragma once
#include "layer.h"

// A multilayer perceptron class 
struct MLP : std::enable_shared_from_this<MLP> {
    public:
    std::vector<std::shared_ptr<Layer>> created_layers {}; // A stack of layers

    // Constructor
    MLP(int inputs, std::vector<int> outputs) {};

    // Compute each layer of the perceptron
    std::shared_ptr<Unit> forward(std::vector<std::shared_ptr<Unit>>& inputs);

    // Set each gradient of the parameters to 0.0
    void zero_grad();

    // Return every trainable parameter of the MLP
    std::vector<std::shared_ptr<Unit>> parameters();

    // Train the MLP with a dataset
    std::vector<std::shared_ptr<Unit>> fit(std::vector<std::vector<std::shared_ptr<Unit>>> inputs,
        std::vector<std::shared_ptr<Unit>> targets, int num_iter, double learning_rate);
};