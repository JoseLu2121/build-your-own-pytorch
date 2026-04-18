#include "tensor.h"
#include <fstream>
#include <stdexcept>
#include "../include/utils.h"
using namespace std;

// ====================
// Constructors & Factory
// ====================

shared_ptr<Tensor> Tensor::create(const vector<int>& shape_param, const vector<float>& data_param,
                                  const vector<shared_ptr<Tensor>>& parents_param) {
    return shared_ptr<Tensor>(new Tensor(shape_param, data_param, parents_param));
}

Tensor::Tensor(const vector<int>& shape_param, const vector<float>& data_param,
       const vector<shared_ptr<Tensor>>& parents_param) 
      // Setting the params given to the attributes of the tensor
    : data(nullptr), 
      shape(shape_param), 
      total_size(element_vector_product(shape_param)), 
      parents(parents_param) {
    
    // If the size is 0 we return an empty tensor
    if (total_size == 0) return;

    // The data is a pointer to a float array with tensor size
    this->data = shared_ptr<float[]>(new float[total_size]);
    
    // We initialize the data with the param given
    if (!data_param.empty()) {
        if (data_param.size() != total_size) {
            throw std::invalid_argument("The data given doesn't match the shape of the tensor");
        }

        for (size_t i = 0; i < total_size; i++) {
            data[i] = data_param[i];
        }

    }else{
        // If we have no data given, we set the tensor data to zero
        for (size_t i = 0; i < total_size; i++) {
            data[i] = 0.0f;
        }
    }

    // We initialize the Strides vector safely avoiding segfaults if shape is empty
    if (!shape.empty()) {
        strides.resize(shape.size());
        // The last one is always 1
        strides.back() = 1;
        // Each element is himself times the previous element from the right
        for (int i = (int)shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}

// Copy or view constructor
Tensor::Tensor(const Tensor& other) 
    : data(other.data),       
      shape(other.shape),     
      strides(other.strides), 
      total_size(other.total_size),
      grad(nullptr), // We avoid issues with the original grad tensor
      parents({}){}


// ====================
// Utils for debugging
// ====================

void Tensor::printElements(int count) const {
    cout << "Elementos del tensor" << endl;
    for (int i = 0; i < count; i++) {
        cout << "Elemento " << i << ": " << getData()[i] << endl;
    }
}

void Tensor::printShape() const{ 
    cout << "Shape: ("; 
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i != shape.size() - 1) cout << ", ";
    }
    cout << ")" << endl;
}

void Tensor::printStrides() const { 
    cout << "Strides: ("; 
    for (size_t i = 0; i < strides.size(); i++) {
        cout << strides[i];
        if (i != strides.size() - 1) cout << ", ";
    }
    cout << ")" << endl;
}

void Tensor::info(int max_size) const {
    cout << "Tensor Info:" << endl;
    cout << "  Shape: (";
    for (size_t i = 0; i < shape.size(); i++) 
        cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    cout << ")" << endl;
    
    cout << "  Strides: (";
    for (size_t i = 0; i < strides.size(); i++) 
        cout << strides[i] << (i < strides.size() - 1 ? ", " : "");
    cout << ")" << endl;

    // We only print the elements if the tensor is small 
    if (total_size <= (size_t)max_size) {
        cout << "  Data: [ ";
        for (int i = 0; i < max_size; i++) cout << data[i] << " ";
        cout << "]" << endl;
    } else {
        cout << "  Data: [ ... " << total_size << " elementos ... ]" << endl;
    }
    cout << "-------------------------" << endl;
}

shared_ptr<Tensor> Tensor::zeros(const vector<int>& shape) {
    return Tensor::create(shape, vector<float>(element_vector_product(shape),0.0f));
}

shared_ptr<Tensor> Tensor::ones(const vector<int>& shape) {
    return Tensor::create(shape, vector<float>(element_vector_product(shape),1.0f));
}

shared_ptr<Tensor> Tensor::random(const vector<int>& shape, float min_val, float max_val) {
    size_t size = element_vector_product(shape);
    vector<float> random_data(size);
    
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);
    for (auto& val : random_data) val = dis(gen);

    return Tensor::create(shape, random_data);
}