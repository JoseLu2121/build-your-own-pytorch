#pragma once
#include "neuron.h"
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>

// A layer of neurons class
struct Layer : std::enable_shared_from_this<Layer> {
    public:
    std::vector<std::shared_ptr<Neuron>> neurons {}; // List of neurons

    // Constructor
    Layer(int inputs, int out);

    // Compute all the neurons
    std::vector<std::shared_ptr<Unit>> forward(std::vector<std::shared_ptr<Unit>>& inputs);

    // Get all the trainable parameters of the layer
    std::vector<std::shared_ptr<Unit>> parameters();
};
