/* pdz_c.cpp
Author: Huaiyu Duan (UNM)
Description: C++ implementation of pfd which approximates the first order derivatives using the central finite difference. 
*/
#include <tuple>
#include "pdz.hpp"

using namespace pybind11::literals;

// python wrapper for pdz
pybind11::cpp_function pdz(nparray x, const std::string &method = "fd5") {
    return pybind11::cpp_function(ctrdiff::pdz(x, method));
}


// expose pdz() to Python
PYBIND11_MODULE(pdz_c, m) {
    m.attr("DIFFERENTIATORS") = pybind11::cast(std::make_tuple("fd3", "fd5", "fd7", "fd9")); // number of points that pfd can use
    m.def("pdz", &pdz, "First order derivative of y by x using n-point central differencing.", "x"_a, "method"_a="fd5");
}