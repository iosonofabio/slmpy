#include <map>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <slmpy.h>


void Network::fromPython(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > edgesIn,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > nodesIn,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > clustersIn,
    uint64_t nClusters) {

    // Make empty communities
    for(uint64_t clusterId=0; clusterId < nClusters; clusterId++) {
        Cluster c(clusterId);
        clusters.push_back(c);
    }

    // Fill the nodes and communities
    uint64_t tmp;
    uint64_t tmp2;
    uint64_t nNodesDone = 0;
    for(uint64_t nodeId=0; nodeId < nNodes; nodeId++) {
        Node n(nodesIn(nodeId, 0));
        n.cluster = clustersIn(nNodesDone++, 0);
        nodes.push_back(n);
    }

    // Fill the edges
    for(uint64_t edgeId=0; edgeId < nEdges; edgeId++) {
        tmp = edgesIn(edgeId, 0);
        tmp2 = edgesIn(edgeId, 1);
        for(std::vector<Node>::iterator n=nodes.begin();
            n != nodes.end();
            n++) {
            if(n->nodeId == tmp) {
                n->neighbors.push_back(tmp2);
                break;
            }
         }
        for(std::vector<Node>::iterator n=nodes.begin();
            n != nodes.end();
            n++) {
            if(n->nodeId == tmp2) {
                n->neighbors.push_back(tmp);
                break;
            }
         }
    }

    // Fill the clusters
    for(std::vector<Node>::iterator n=nodes.begin();
        n != nodes.end();
        n++) {
        clusters[n->cluster].nodes.push_back(*n);
    }
}
