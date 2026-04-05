#include <iostream>
#include <cmath>

double loss_function(double input, double w, double y_target) {
    double y_pred = input * w;
    return std::pow(y_pred - y_target, 2);
}

int main() {
    std::cout << "--- Training a single neuron ---\n\n";

    // Initial values
    double input = 2.0;
    double w = 1.0;
    double y_target = 10.0;
    double learning_rate = 0.1;

    // Forward initial pass
    double initial_loss = loss_function(input, w, y_target);
    std::cout << "1. Initial state:\n";
    std::cout << "   w = " << w << ", y_pred = " << (input * w) << "\n";
    std::cout << "   Initial loss = " << initial_loss << "\n\n";

    // We calculate the derivative
    double h = 0.0001;
    double loss_with_h = loss_function(input, w + h, y_target);
    double gradient = (loss_with_h- initial_loss) / h;

    std::cout << "2. We calculate the gradient:\n";
    std::cout << "   Gradient: " <<  gradient  << "\n";

    // We optimize the weight using gradient descent
    std::cout << "3. Optimizing using gradient descent:\n";
    std::cout << "   Formula: w = w - (learning_rate * gradient)\n";
    w = w - (learning_rate * gradient);
    std::cout << "   New weight w = " << w << "\n\n";

    // Loss function after optimizing
    double final_loss = loss_function(input, w, y_target);
    std::cout << "Loss function after optimizing:\n";
    std::cout << "   w = " << w << ", y_pred = " << (input * w) << "\n";
    std::cout << "   Final loss = " << final_loss << "\n";

    return 0;
}