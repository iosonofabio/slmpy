//#include <iostream>
#include <set>
#include <map>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#ifndef SLMPY_VERBOSE
#define SLMPY_VERBOSE false
#endif

namespace py = pybind11;

class Node {
    public:
        uint64_t nodeId;
        std::map<uint64_t, double> neighbors;
        uint64_t cluster;

        Node() {};
        Node(uint64_t nId) {nodeId = nId;};
        Node(uint64_t nId, uint64_t clId) {nodeId = nId; cluster = clId;};
        Node(uint64_t nId, uint64_t clId, std::map<uint64_t, double>neighIds) {nodeId = nId; cluster = clId; neighbors = neighIds;};

        double degree();
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
        std::map<uint64_t, Node> nodes;
        std::vector<Cluster> clusters;
        std::set<uint64_t> fixedNodes;

        Network() {};

        void fromPython(
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edgesIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > clustersIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > fixedNodesIn);
        void toPython(
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
            py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communitiesOut);

        double calcTwiceTotalEdges();
        void calcClustersFromNodes();
        double calcModularity();
        std::vector<uint64_t> nodesInRadomOrder(uint32_t seed);
        uint64_t findBestCluster(uint64_t nodeId);
        void updateCluster(uint64_t nodeId, uint64_t clusterId);

        void createSingletons();
        void createFromSubnetworks(std::vector<Network> subnetworks);
        void mergeClusters(std::vector<Cluster> clusters);
        Network calculateReducedNetwork();
        std::vector<Network> createSubnetworks();

        bool runLocalMovingAlgorithm(uint32_t randomSeed);
        bool runLouvainAlgorithm(uint32_t randomSeed);
        bool runSmartLocalMovingAlgorithm(uint32_t randomSeed, int64_t maxIterations = -1);
};
