#include <iostream>
#include <cmath>

int main() {
    std::cout << "--- Chain rule and Backpropagation ---\n\n";

    // Initial values
    double input = 2.0;
    double w1 = 3.0;
    double b = 1.0;
    double w2 = 2.0;
    double y_target = 10.0;

    std::cout << "1. Forward pass:\n";
    
    // Hidden neuron
    double h = (input * w1) + b;
    std::cout << "   h = input * w1 + b = " << h << "\n";
    
    // Output neuron
    double y_pred = h * w2;
    std::cout << "   y_pred = h * w2 = " << y_pred << "\n";

    // Loss function
    double loss = std::pow(y_pred - y_target, 2);
    std::cout << "   Loss = (y_pred - y_target)^2 = " << loss << "\n\n";

    std::cout << "2. Backward pass:\n";
    
    // Gradient of the output -> L with respect to y_pred
    double dL_dy = 2 * (y_pred - y_target);
    std::cout << "   dL/dy = " << dL_dy << "\n";

    // Propagating to the hidden neuron -> L with respect to h
    double dy_dh = w2;
    double dL_dh = dL_dy * dy_dh;
    std::cout << "   dL/dh = dL/dy * dy/dh = " << dL_dy << " * " << dy_dh << " = " << dL_dh << "\n";

    // Propagating to the weight w1 -> L with respect to w1
    double dh_dw1 = input;
    double dL_dw1 = dL_dh * dh_dw1;
    std::cout << "   dL/dw1 = dL/dh * dh/dw1 = " << dL_dh << " * " << dh_dw1 << " = " << dL_dw1 << "\n\n";

    std::cout << "3. Final result:\n";
    std::cout << "   The gradient of w1 is: " << dL_dw1 << "\n";

    return 0;
}