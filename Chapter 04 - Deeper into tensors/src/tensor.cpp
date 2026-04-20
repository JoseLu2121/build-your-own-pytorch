#include "../include/tensor.h"
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
    size_t virtual_size = element_vector_product(shape);
    if (virtual_size <= (size_t)max_size) {
        cout << "  Data: [ ";
        std::vector<int> current_indices(shape.size(), 0);
        for (size_t i = 0; i < virtual_size; i++) {
            int raw_idx = 0;
            for (size_t d = 0; d < shape.size(); d++) {
                raw_idx += current_indices[d] * strides[d];
            }
            cout << data[raw_idx] << " ";
            
            for (int d = (int)shape.size() - 1; d >= 0; d--) {
                current_indices[d]++;
                if (current_indices[d] < shape[d]) {
                    break;
                }
                current_indices[d] = 0;
            }
        }
        cout << "]" << endl;
    } else {
        cout << "  Data: [ ... " << virtual_size << " elements ... ]" << endl;
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

// ===================
// Size and view manipulation
// ===================

shared_ptr<Tensor> Tensor::reshape(const vector<int>& new_shape) {
    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    size_t new_total_size = element_vector_product(new_shape);
    if (new_total_size != this->total_size) throw std::runtime_error("The new tensor shape needs to have " + 
        std::to_string(this->total_size) + " elements");

    view->shape = new_shape;

    if (new_shape.empty()) {
        view->strides.clear();
        return view;
    }

    // We initialize the Strides vector
    view->strides.resize(new_shape.size());
    // The last one is always 1
    view->strides.back() = 1;
    // Each element is himself times the previous element from the right
    for (int i = (int)view->shape.size() - 2; i >= 0; --i) {
        view->strides[i] = view->strides[i + 1] * view->shape[i + 1];
    }

    return view;
}

// Get the batch related to an index
shared_ptr<Tensor> Tensor::batch_view(int index, bool keep_dim) {
    // Tensor must have three dimensions
    if(this->getDimension() < 3) throw std::runtime_error("Tensor must have 3 dimension in order to create a batch view");

    int n_batch = this->shape[0]; // We pick the batch dim
    if(index < 0 || index >= n_batch) throw std::runtime_error("Batch index is out of bounds");

    auto view = std::shared_ptr<Tensor>(new Tensor(*this)); // It creates a copy of the tensor
    int offset = index * this->strides[0]; // Calculate the index of the first element of the batch

    view->data = std::shared_ptr<float[]>(this->data, this->data.get() + offset); // We create a pointer to a float vector to
                                                                                  // the batch
    if (!keep_dim) { // Keep dim represents if we delete the batch dimension or set it to 1
        view->shape.erase(view->shape.begin());
        view->strides.erase(view->strides.begin());
    }
    else {
        view->shape[0] = 1;
        view->strides[0] = 0;
    }

    view->total_size = element_vector_product(view->shape); // set new total size
    view->parents = {shared_from_this()}; // the parents are the ones in the original tensor

    return view;
}

// Turn 1D and 2D tensors to 3D
shared_ptr<Tensor> Tensor::view_to_3d() {
    // We create a copy of the tensor
    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    // Depending of the dimension we adapt shapes and strides
    if(view->getDimension() == 1){
        view->strides.push_back(0); 
        view->shape.push_back(1);
    }
    
    if(view->getDimension() == 2){
        view->strides.insert(view->strides.begin(), 0); 
        view->shape.insert(view->shape.begin(), 1);
    }
    return view;
}

shared_ptr<Tensor> Tensor::view_to_gemm(bool as_b_term) {
    
    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    // Same logic as view_to_3d, but depends on whether the tensor is the second operand
    if(view->getDimension() == 1){
        if(as_b_term){
            view->shape = {1, view->shape[0], 1};
            view->strides = {0, view->strides[0], 0};
        } else{
            view->shape = {1,1,view->shape[0]};
            view->strides = {0,0, view->strides[0]};
        }
    }
    else if(view->getDimension() == 2){
        view->shape.insert(view->shape.begin(), 1);
        view->strides.insert(view->strides.begin(), 0); 
    }
    else {
        if(view->shape[0] == 1) { view->strides[0] = 0;}
    }
    return view;

}

// returns the out dimension that a tensor must have in a binary operation
vector<int> Tensor::broadcast_shapes(const vector<int>& shape_a, const vector<int>& shape_b) {

    // we calculate the number of dims that out must have
    int max_dims = std::max(shape_a.size(), shape_b.size());
    vector<int> out_shape(max_dims);
    
    // we create dimension index
    int i_a = (int)shape_a.size() - 1;
    int i_b = (int)shape_b.size() - 1;
    int i_out = max_dims - 1;
    
    // iterate from right to left
    while (i_out >= 0) {
        // if any dim is already below zero we set it to one
        int dim_a = (i_a >= 0) ? shape_a[i_a] : 1;
        int dim_b = (i_b >= 0) ? shape_b[i_b] : 1;
        // only compatible if dims are equal or one
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("Tensors cannot be broadcasted together");
        } else {
            // the out dim is the greater one
            out_shape[i_out] = std::max(dim_a, dim_b);
        }
        
        i_a--; i_b--; i_out--;
    }
    return out_shape;
}

shared_ptr<Tensor> Tensor::broadcast_to(const vector<int>& target_shape) {
    if (this->shape == target_shape) return shared_from_this();

    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    view->shape = target_shape;
    view->strides.resize(target_shape.size());
    
    int offset = (int)target_shape.size() - (int)this->shape.size();
    
    for (size_t i = 0; i < target_shape.size(); ++i) {
        int original_idx = (int)i - offset;
        
        if (original_idx < 0) {
            // New dimension (broadcasted) -> stride 0
            view->strides[i] = 0;
        } else {
            // Existing dimension
            if (this->shape[original_idx] == target_shape[i]) {
                view->strides[i] = this->strides[original_idx];
            } else {
                // Dimension 1 stretched -> stride 0
                view->strides[i] = 0;
            }
        }
    }
    
    return view;
}