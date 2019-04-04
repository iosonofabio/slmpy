#include <iostream>
#include <map>
#include <cmath>
#include <cstdlib>   // std::rand, std::srand
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <slmpy.h>

namespace py = pybind11;


///////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////
int local_moving(
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communities_out,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edges,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodes,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > communities,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > fixedNodes,
    uint32_t random_seed,
    uint64_t n_iterations) {

    std::srand(random_seed);

    Network net;
    net.fromPython(edges, nodes, communities, fixedNodes);
    for(uint64_t i=0; i<n_iterations; i++)
        net.runLocalMovingAlgorithm(random_seed);
    net.toPython(nodes, communities_out);

    return 0;
};


int louvain(
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communities_out,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edges,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodes,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > communities,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > fixedNodes,
    uint32_t random_seed) {

    std::srand(random_seed);

    Network net;
    net.fromPython(edges, nodes, communities, fixedNodes);
    net.runLouvainAlgorithm(random_seed);
    net.toPython(nodes, communities_out);

    return 0;
};

int smart_local_moving(
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communities_out,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edges,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodes,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > communities,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > fixedNodes,
    uint32_t random_seed,
    uint64_t n_iterations) {

    std::srand(random_seed);

    Network net;
    net.fromPython(edges, nodes, communities, fixedNodes);
    for(uint64_t i=0; i != n_iterations; i++)
        net.runSmartLocalMovingAlgorithm(random_seed);
    net.toPython(nodes, communities_out);

    return 0;
};


PYBIND11_MODULE(_slmpy, m) {
    m.def("local_moving", &local_moving, R"pbdoc(
        Local Moving algorithm in C++, exported to
        Python via pybind11.
    )pbdoc");

    m.def("louvain", &louvain, R"pbdoc(
        Louvain algorithm in C++, exported to
        Python via pybind11.
    )pbdoc");

    m.def("smart_local_moving", &smart_local_moving, R"pbdoc(
        Smart Local Moving algorithm in C++, exported to
        Python via pybind11.
    )pbdoc");
};

