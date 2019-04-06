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
        std::map<uint64_t, double> neighbors;
        uint64_t cluster;
        double degree;
        double degreeGlobal;

        Node() {};
        Node(uint64_t clId) {cluster = clId;};
        Node(uint64_t clId, std::map<uint64_t, double>neighIds) {cluster = clId; neighbors = neighIds;};

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
        std::map<uint64_t, Node> nodes;
        std::vector<Cluster> clusters;
        std::set<uint64_t> fixedNodes;
        double twiceTotalEdges;
        bool isSubnetwork = false;
        double twiceTotalEdgesGlobal;

        Network() {};

        void fromPython(
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edgesIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > clustersIn,
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > fixedNodesIn);
        void toPython(
            py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
            py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communitiesOut);

        void calcDegreesAndTwiceTotalEdges();
        void calcClustersFromNodes();
        //double calcModularity();
        std::vector<uint64_t> nodesInRadomOrder();
        uint64_t findBestCluster(uint64_t nodeId);
        void updateCluster(uint64_t nodeId, uint64_t clusterId);

        void createSingletons();
        void createFromSubnetworks(std::map<uint64_t, uint64_t> clusterToSubnetwork);
        void mergeClusters(std::vector<Cluster> clusters);
        Network calculateReducedNetwork();
        Network createSubnetwork(std::vector<Cluster>::iterator c);

        bool runLocalMovingAlgorithm();
        bool runLouvainAlgorithm();
        bool runSmartLocalMovingAlgorithm();

        unsigned seed = 0;

};
