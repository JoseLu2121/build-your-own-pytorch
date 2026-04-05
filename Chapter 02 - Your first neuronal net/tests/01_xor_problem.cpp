#include <iostream>
#include <vector>
#include <memory>
#include "../include/unit.h"
#include "../include/mlp.h"
using namespace std;

int main() {
    cout << "--- Starting XOR training ---\n" << endl;

    // Out inputs
    vector<vector<shared_ptr<Unit>>> X = {
        {make_shared<Unit>(0.0, "x1"), make_shared<Unit>(0.0, "x2")},
        {make_shared<Unit>(0.0, "x1"), make_shared<Unit>(1.0, "x2")},
        {make_shared<Unit>(1.0, "x1"), make_shared<Unit>(0.0, "x2")},
        {make_shared<Unit>(1.0, "x1"), make_shared<Unit>(1.0, "x2")}
    };

    // We set out targets
    vector<shared_ptr<Unit>> Y = {
        make_shared<Unit>(0.0, "y1"),
        make_shared<Unit>(1.0, "y2"),
        make_shared<Unit>(1.0, "y3"),
        make_shared<Unit>(0.0, "y4")
    };

    // We define an MLP with 1 hidden layer
    vector<int> architecture = {4, 1};
    MLP model(2, architecture);

    // We train out model on 10000 epochs
    int epochs = 10000;
    double learning_rate = 0.01;
    
    cout << "Training model on " << epochs << " epoch and learning rate " << learning_rate << "...\n" << endl;
    
    // We call the fit function to train
    vector<shared_ptr<Unit>> predictions = model.fit(X, Y, epochs, learning_rate);

    // We check if the model has trained well
    cout << "\n--- Final results ---" << endl;
    cout << "Input (0, 0) -> Prediction: " << predictions[0]->data << " | Waiting: 0" << endl;
    cout << "Input (0, 1) -> Prediction: " << predictions[1]->data << " | Waiting: 1" << endl;
    cout << "Input (1, 0) -> Prediction " << predictions[2]->data << " | Waiting: 1" << endl;
    cout << "Input (1, 1) -> Prediction: " << predictions[3]->data << " | Waiting: 0" << endl;

    return 0;
}