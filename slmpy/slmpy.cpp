//#include <iostream>
#include <map>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <slmpy.h>

namespace py = pybind11;

//using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


// Fill knn output matrix
void fillOutputVector(
    std::vector<uint64_t> communities_from_network,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > communities_out) {

    uint64_t irow = 0;
    for(std::vector<long unsigned int>::iterator c=communities_from_network.begin();
        c != communities_from_network.end();
        c++) {
        communities_out(irow, 1) = *c;
    }
}



///////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////
int smart_local_moving(
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > communities_out,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > edges,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > nodes,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > communities,
    uint32_t random_seed,
    uint64_t n_iterations) {

    uint64_t nNodes = nodes.rows();
    uint64_t nEdges = edges.rows();

    Network net(nNodes, nEdges);
    net.fromPython(edges, nodes, communities);

    for(uint64_t it=0; it < n_iterations; it++) {
        net.runLocalMovingAlgorithm(random_seed);
    }
    fillOutputVector(net.getClusterIds(), communities_out);

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

