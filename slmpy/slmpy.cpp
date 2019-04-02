//#include <iostream>
#include <map>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <slmpy.h>

namespace py = pybind11;

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


//// Fill knn output matrix
//void fillOutputVector(
//    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > communities_out,
//    ) {
//
//    uint64_t irow = 0;
//    for(std::vector<long unsigned int>::iterator c=communities.begin();
//        c != communities.end();
//        c++) {
//        communities_out(irow) = c;
//    }
//}
//

//double computeModularity(
//    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > edges,
//    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nodes,
//    ) {
//
//    uint64_t nE = edges.size();
//    uint64_t nN = nodes.size();
//
//    double m = 0;
//
//
//}


///////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////
int smart_local_moving(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > edges,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > nodes,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > communities,
    uint64_t n_nodes,
    uint64_t n_edges,
    uint64_t n_communities,
    uint32_t random_seed,
    uint64_t n_iterations) {

    Network net(n_nodes, n_edges);
    net.fromPython(edges, nodes, communities, n_communities);

    for(uint64_t it=0; it < n_iterations; it++) {
        net.runLocalMovingAlgorithm(random_seed);
    }

    return (int) 100 * net.calcModularity();
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

