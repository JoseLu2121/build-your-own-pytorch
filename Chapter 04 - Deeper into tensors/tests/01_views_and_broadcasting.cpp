#include "../include/tensor.h"
#include "../include/utils.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    cout << "--- 1. Views and Reshape ---" << endl;
    cout << "Original 1D Tensor:" << endl;
    
    auto tensor_1d = Tensor::create({6}, {1, 2, 3, 4, 5, 6});
    tensor_1d->info(6);

    cout << "Creating a 2D View (2, 3) from 1D Tensor:" << endl;
    auto tensor_2d = tensor_1d->reshape({2, 3});
    tensor_2d->info(6);

    if (tensor_1d->getData() == tensor_2d->getData()) {
        cout << " Data was reshape successfully" << endl;
    }

    cout << "\n--- 2. Converting to 3D View ---" << endl;
    cout << "Converting 2D View to 3D View for efficient operations:" << endl;
    auto tensor_3d = tensor_2d->view_to_3d();
    tensor_3d->info(6);

    cout << "\n--- 3. Broadcasting Mechanism ---" << endl;
    cout << "Original Vector (1D):" << endl;
    auto vector_1d = Tensor::create({3}, {10, 20, 30});
    vector_1d->info(3);

    cout << "Target Matrix Shape: (3, 3)" << endl;
    vector<int> target_shape = {3, 3};
    
    auto result_shape = Tensor::broadcast_shapes(vector_1d->getShape(), target_shape);
    
    auto broadcasted_view = vector_1d->broadcast_to(result_shape);
    broadcasted_view->info(9);
    
    cout << "Notice how the stride for the expanded dimension is 0!" << endl;
    cout << "This allows reading the same physical memory elements without allocating new ones." << endl;

    return 0;
}
