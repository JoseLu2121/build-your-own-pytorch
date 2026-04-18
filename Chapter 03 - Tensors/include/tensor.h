#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <unordered_set>
#include <random>

// tensor class 
class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    // data storage 
    std::shared_ptr<float[]> data;
    // number of elements of the tensor
    size_t total_size;
    // shape of the tensor (batch,row,col)
    std::vector<int> shape;
    // strides for each dimension
    std::vector<int> strides;
    
    // parent nodes of the tensor in the computation graph
    std::vector<std::shared_ptr<Tensor>> parents;
    // gradient tensor
    std::shared_ptr<Tensor> grad;
    // backward function
    std::function<void()> _backward;

    // constructor
    Tensor(const std::vector<int>& shape_param, const std::vector<float>& data_param = {},
           const std::vector<std::shared_ptr<Tensor>>& parents_param = {});
    
    // copy constructor
    Tensor(const Tensor& other);

public:
    // creates a tensor safely managed by a shared_ptr
    static std::shared_ptr<Tensor> create(const std::vector<int>& shape_param, 
                                          const std::vector<float>& data_param = {},
                                          const std::vector<std::shared_ptr<Tensor>>& parents_param = {});

    // all elements are 0
    static std::shared_ptr<Tensor> zeros(const std::vector<int>& shape);
    
    // all elements are 1
    static std::shared_ptr<Tensor> ones(const std::vector<int>& shape);
    
    // elements are random values between min_val and max_val
    static std::shared_ptr<Tensor> random(const std::vector<int>& shape, float min_val = -1.0f, float max_val = 1.0f);

    // get the size of the tensor
    size_t getSize() const { return total_size; }
    
    // get the dimension of the tensor
    int getDimension() const { return shape.size(); }
    
    // get the pointer to the data of the tensor
    float* getData() const { return data.get(); }
    
    // get the gradient tensor
    std::shared_ptr<Tensor> getGrad() const { return grad; }
    
    // get the vector of strides of the tensor
    const std::vector<int>& getStrides() const { return strides; }
    
    // get the vector of the shape of the tensor
    const std::vector<int>& getShape() const { return shape; }
    
    // get the parents of the tensor
    const std::vector<std::shared_ptr<Tensor>>& getParents() const { return parents; }

    void set_backward(std::function<void()> bw) { _backward = bw; }
    
    void set_parents(const std::vector<std::shared_ptr<Tensor>>& p) { parents = p; }

    void set_shape(const std::vector<int>& s) { shape = s; }

    void set_strides(const std::vector<int>& s) { strides = s; }

    void init_grad() { if (!grad) grad = Tensor::zeros(shape); }

    // compute backpropagation
    void backward();

    // prints the first n elements of the tensor
    void printElements(int count = 1) const;

    // prints the shape of the tensor
    void printShape() const;

    // prints the strides of the tensor
    void printStrides() const;

    // prints all the info of the tensor with a max number of data elements
    void info(int max_size = 20) const;

};