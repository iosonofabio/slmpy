#include <iostream>
#include <map>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <slmpy.h>

namespace py = pybind11;


///////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////
int smart_local_moving(
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communities_out,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edges,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodes,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > communities,
    uint32_t random_seed,
    uint64_t n_iterations) {

    Network net;
    net.fromPython(edges, nodes, communities);

    for(uint64_t it=0; it < n_iterations; it++) {
        net.runLocalMovingAlgorithm(random_seed);
    }
    net.toPython(communities_out);

    return 0;
};

PYBIND11_MODULE(_slmpy, m) {
    m.def("smart_local_moving", &smart_local_moving, R"pbdoc(
        Smart Local Moving algorithm in C++, exported to
        Python via pybind11.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
};

