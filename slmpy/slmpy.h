//#include <iostream>
#include <map>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class Node {
    public:
        int nodeId;
        std::vector<uint64_t> neighbors;
        int cluster;
};



// this is the interface fuction
int smart_local_moving(
    // I make copies of the data structures in C++ for convenience
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > edges,        
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > nodes,        
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > communities,
    uint64_t n_iterations);
