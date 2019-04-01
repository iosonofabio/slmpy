//#include <iostream>
#include <map>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class Node {
    public:
        uint64_t nodeId;
        std::vector<uint64_t> neighbors;
        uint64_t cluster;
        uint64_t degree();

        Node(uint64_t nId) {nodeId = nId;};
};

class Cluster {
    public:
        uint64_t clusterId;
        std::vector<Node> nodes;
        double nEdges;

        Cluster(uint64_t clusterId) {this->clusterId = clusterId;};

};

class Network {
    public:
        uint64_t nNodes;
        uint64_t nEdges;
        std::vector<Node> nodes;
        std::vector<Cluster> clusters;

        Network(uint64_t nN, uint64_t nE) {nNodes = nN; nEdges = nE;};

        void fromPython(
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > edgesIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > nodesIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > clustersIn,
            uint64_t nClusters);

};

// this is the interface fuction
int smart_local_moving(
    // I make copies of the data structures in C++ for convenience
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > edges,        
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > nodes,        
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > communities,
    uint64_t n_nodes,
    uint64_t n_edges,
    uint64_t n_communities,
    uint64_t n_iterations);
