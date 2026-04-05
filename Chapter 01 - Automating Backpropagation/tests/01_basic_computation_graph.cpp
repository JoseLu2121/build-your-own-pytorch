#include "../include/ops.h"
#include "../include/unit.h"

using namespace std;

int main() {
    // Initial Units
    auto a = make_shared<Unit>(2.0,"a");
    auto b = make_shared<Unit>(3.0,"b");
    auto c = make_shared<Unit>(4.0, "c");

    // Operations between Units
    auto d = a + b;
    d->label = "d";
    auto e = d * c;
    e->label = "e";

    // We propragate the gradient
    e->retropropagate();

    vector<shared_ptr<Unit>> units = {a, b, c, d, e};

    cout << "Units gradients:" << endl;
    for (const auto& u : units) {
        cout << "Node " << u->label 
             << " | data: " << u->data 
             << " | grad: " << u->grad << "\n";
    }

    return 0;
};