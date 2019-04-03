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

        Node() {};
        Node(uint64_t nId) {nodeId = nId;};
        Node(uint64_t nId, uint64_t clId) {nodeId = nId; cluster = clId;};
        Node(uint64_t nId, uint64_t clId, std::vector<uint64_t>neighIds) {nodeId = nId; cluster = clId; neighbors = neighIds;};

        uint64_t degree();
};

class Cluster {
    public:
        uint64_t clusterId;
        std::vector<uint64_t> nodes;

        Cluster(uint64_t clId) {clusterId = clId;};
        Cluster(uint64_t clId, std::vector<uint64_t> ns) {clusterId = clId; nodes = ns;};

};

class Network {
    public:
        uint64_t nNodes;
        uint64_t nEdges;
        std::map<uint64_t, Node> nodes;
        std::vector<Cluster> clusters;

        Network() {};

        void fromPython(
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edgesIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > clustersIn);
        void toPython(
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
            py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communitiesOut);

        std::vector<uint64_t> nodesInRadomOrder(uint32_t seed);
        bool runLocalMovingAlgorithm(uint32_t randomSeed, int64_t maxIterations = -1);
        bool runSmartLocalMovingAlgorithm(uint32_t randomSeed, int64_t maxIterations = -1);

        double calcModularity();
        uint64_t findBestCluster(uint64_t nodeId);
        void updateCluster(uint64_t nodeId, uint64_t clusterId);

};
