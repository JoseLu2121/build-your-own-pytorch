#pragma once

#include <memory>
#include "unit.h"

// We include each Unit operation (check ops.cpp for a more detailed explanation)
std::shared_ptr<Unit> operator+(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> operator-(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> operator*(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> operator/(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> operatorpow(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> relu(const std::shared_ptr<Unit>& a);
std::shared_ptr<Unit> operator_tanh(const std::shared_ptr<Unit>& a);