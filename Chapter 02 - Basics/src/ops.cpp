#include "ops.h"
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>

using namespace std;

// This file include each Unit operation, when you add the prefix operator before 
// +, -, * or /, your replacing the basic operation of c++ with the function you
// define, so you can make things like Unit + Unit

// Add operation
shared_ptr<Unit> operator+(const shared_ptr<Unit>& a, const shared_ptr<Unit>& b) {
    // The output is a new Unit with a value of a + b
    auto out = make_shared<Unit>(a->data + b->data, vector<shared_ptr<Unit>>{a, b}); // We add a and b as out childrens

    // The local derivative of (a + b) with respect to 'a' or 'b' are both 1, so we just apply
    // the chain rule -> 1 * out.grad
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
};

// Mult operation
shared_ptr<Unit> operator*(const shared_ptr<Unit>& a, const shared_ptr<Unit>& b) {
    // The output is simply a*b
    auto out = make_shared<Unit>(a->data * b->data, vector<shared_ptr<Unit>>{a,b}); // We add a and b as out childrens

    // The local derivative of (a * b) with respect to 'a' is b, and with respect to 'b' is a
    // So we apply the chain rule a.grad = b * out.grad and viceversa 
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad * b->data;
        b->grad += out->grad * a->data;
    };
    return out;
};

// Sub operation
shared_ptr<Unit> operator-(const shared_ptr<Unit>& a, const shared_ptr<Unit>& b) {
    // The output is a - b
    auto out = make_shared<Unit>(a->data - b->data, vector<shared_ptr<Unit>>{a,b});

    // The derivate is the same as add, but the second operand local derivate is negative
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad;
        b->grad += -out->grad;
    };
    return out;
};

// Pow operation
shared_ptr<Unit> operatorpow(const shared_ptr<Unit>& a, double b){
    // We use the pow cmath C++ function 
    auto out = make_shared<Unit>(pow(a->data, b), vector<shared_ptr<Unit>>{a});

    // The derivative of a^b, with respect to a is just b * a^(b-1)
    out->backward = [a,b,out] () mutable {
        a->grad += b * pow(a->data,(b - 1)) * out->grad;
    };
    return out;
}

// Div operation
shared_ptr<Unit> operator/(const shared_ptr<Unit>& a, const shared_ptr<Unit>& b) {
    // We can make a div operation using mult and pow -> (a/b) = a * b^-1
    return a * operatorpow(b, -1);
}


// Relu operation
shared_ptr<Unit> relu(const shared_ptr<Unit>& a) {
    // If the value is less than zero, we set it to zero
    double t = a->data < 0 ? 0.0 : a->data;
    auto out = make_shared<Unit>(t, vector<shared_ptr<Unit>>{a});

    // The derivative of relu is 0 or 1 depending if the value is greater or lower than 0,
    // so out.data > 0 ? 1 * out.grad : 0 * out.grad (chain rule)
    out->backward = [a,out] () mutable {
        a->grad += (out->data > 0) ? out->grad : 0.0;
    };
    return out;
};

// Tanh operator
shared_ptr<Unit> operator_tanh(const shared_ptr<Unit>& a) {
    // we use the cmath C++ library
    auto output_data = tanh(a->data);
    auto out = make_shared<Unit>(output_data, vector<shared_ptr<Unit>>{a});

    // the derivative of tanh(x) = 1 - tanh²(x)
    out->backward = [a,out] () mutable {
        double t2 = tanh(a->data);
        a->grad += (1 - t2*t2) * out->grad;
    };
    return out;
}