#include "mlp.h"

using namespace std;

// Implementation of the class defined in include/mlp.h


MLP::MLP(int inputs, vector<int> outputs){
    vector<int> layers {};
    // First we add the input number of each layer
    layers.push_back(inputs);
    for(auto& l: outputs){
        layers.push_back(l);
    }

    // We iterate to create layers with the inputs and outputs correctly
    for(size_t i = 0; i<layers.size()-1;i++){
        auto layer = make_shared<Layer>(layers.at(i),layers.at(i+1));
        created_layers.push_back(layer);
    }
}

// Forward pass of the entire MLP
shared_ptr<Unit> MLP::forward(vector<shared_ptr<Unit>>& inputs){
    vector<shared_ptr<Unit>> layer_output = inputs;
    // We just forward each layer with the output of the previous one
    for (auto& layer : this->created_layers){
        layer_output = layer->forward(layer_output);
    }
    // Return the last layer output
    return layer_output.at(0);
}

// Set every parameter gradient to zero
void MLP::zero_grad(){
    for(auto& p : this->parameters()){
        p->grad = 0;
    }
}

// Return every trainable parameter of the MLP
vector<shared_ptr<Unit>> MLP::parameters(){
    vector<shared_ptr<Unit>> out {};
    // The parameters returned will be the weight and bias of each layer
    for(auto& l: this->created_layers){
        for(auto& p : l->parameters()){
            out.push_back(p); 
        }
    }
    return out;
}

// Train the MLP with a dataset
vector<shared_ptr<Unit>> MLP::fit(vector<vector<shared_ptr<Unit>>> inputs,
    vector<shared_ptr<Unit>> targets, int num_iter, double learning_rate) {

    vector<shared_ptr<Unit>> output;
    // For each iteration
    for(int i=0; i < num_iter ; i++){
        output = {};
        shared_ptr<Unit> loss_mse_acum = make_shared<Unit>(0.0,"loss"); // Create a loss value Unit

        // For each row in the dataset
        for(size_t j=0; j<targets.size();j++) {
            // We forward the entire MLP with the current row as the input
            auto o = this->forward(inputs.at(j));
            // Create the loss function, in this case MSE = mean(sum((output - target)^2))
            auto target = targets.at(j);
            auto loss = o - target;
            auto loss_mse = loss * loss;
            loss_mse_acum = loss_mse_acum + loss_mse;
            output.push_back(o);
        }

        // Apply the mean creating an Unit with a value of the number of rows
        loss_mse_acum = loss_mse_acum / make_shared<Unit>(output.size(),vector<shared_ptr<Unit>> {});

        // Before we do backpropagation we need to set each gradient to 0.0
        this->zero_grad();
        // Apply backpropagation from the loss Unit
        loss_mse_acum->backward_total();

        // Update the gradient to the trainable parameters -> weight += -(lr * gradient)
        for(auto& p : this->parameters()){
            p->data += -learning_rate * p->grad;
            
            
        }

        cout  << "Iteration: " << i << " loss= " << loss_mse_acum->data << endl;
        for(auto& p:output){
            cout << p->data << endl;
        }
    }
    return output;
}