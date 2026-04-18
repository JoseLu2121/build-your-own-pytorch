#include "../include/tensor.h"
#include "../include/utils.h"
#include <iostream>

using namespace std;


int main() {

    cout << "Matrix Tensor:" << endl;

    auto tensor_matrix = Tensor::create({2,3},{1,2,3,4,5,6});
    tensor_matrix->info(6);

    auto data_pointer_matrix = tensor_matrix->getData();

    auto coordinate_1_0_matrix = 1 * tensor_matrix->getStrides()[0] + 
    0 * tensor_matrix->getStrides()[1];

    auto data_at_coordinate_matrix = data_pointer_matrix[coordinate_1_0_matrix];

    cout << "Coordinate (1,0) (expected 4): "  << data_at_coordinate_matrix << endl;

    cout << "=======================================" << endl;

    cout << "Batch Tensor:" << endl;

    auto tensor_batch = Tensor::create({3,2,2}, {11,12,13,14,21,22,23,24,31,32,33,34});
    tensor_batch->info(12);

    auto data_pointer_batch = tensor_batch->getData();

    auto coordinate_1_0_batch = 2 * tensor_batch->getStrides()[0] + 
    1 * tensor_batch->getStrides()[1] + 0 * tensor_batch->getStrides()[2];

    auto data_at_coordinate_batch = data_pointer_batch[coordinate_1_0_batch];

    cout << "Coordinate (3,1,0) (expected 33): "  << data_at_coordinate_batch << endl;

    return 0;
}