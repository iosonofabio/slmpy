#include <set>
#include <map>
#include <cmath>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/shuffle_order.hpp>
#include <boost/random/linear_congruential.hpp>

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
        Cluster c(clustersIn(clusterId, 0));
        clusters.push_back(c);
    }

    // Fill the nodes
    uint64_t tmp;
    uint64_t tmp2;
    uint64_t nNodesDone = 0;
    for(uint64_t nodeId=0; nodeId < nNodes; nodeId++) {
        Node n(nodesIn(nodeId, 0), clustersIn(nNodesDone++, 0));
        nodes.push_back(n);
    }

    // Fill the edges (both directions)
    // NOTE: this assumes that the input edges are unique
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
        for(std::vector<Cluster>::iterator c=clusters.begin();
            c != clusters.end();
            c++) {
            if(n->cluster == c->clusterId) {
                c->nodes.push_back(*n);
                break;
            }
        }
    }
}


//// Calculate the modularity
////
//// Q = 1 / 2m * sum_ij [ A_ij - k_i k_j / 2m ] delta(c_i, c_j)
////
//// where:
////  m = # edges
////  A_ij = 1 is ij have an edge, else 0
////  k_i = degree of node i
////
////  In practice, only terms within each community contribute, so we sum over communities instead
////
////  Q = 1 / 2m * sum_c [ sum_ij [ A_ij - k_i k_j / 2m ] ]
////
////  where ij are now only within the community
////
////  sum_ij A_ij is twice the numer of edges within the community
////  sum_ij k_i k_j / 2m = 1 / 2m * (sum_i k_i) (sum_j k_j)
////  sum_i k_i is the total summed degree of the community (including edges that go out of it)
////  Bottomline is we should calculate the following useful quantities:
////   the number of edges within the community
////   the summed degree of the community
double Network::calcModularity() {
    double mod = 0;

    for(std::vector<Cluster>::iterator c=clusters.begin();
        c != clusters.end();
        c++) {

        // Calculate twice the number of edges within the community
        for(std::vector<Node>::iterator n=c->nodes.begin();
            n != c->nodes.end();
            n++) {
            for(std::vector<uint64_t>::iterator neiId=n->neighbors.begin();
                neiId != n->neighbors.end();
                neiId++) {
                for(std::vector<Node>::iterator n2=c->nodes.begin();
                    n2 != c->nodes.end();
                    n2++) {
                    if(n2->nodeId == *neiId) {
                        mod += 1;
                        break;
                    }
                }
            }
        }

        // Subtract the summed degrees
        double sumDeg = 0;
        for(std::vector<Node>::iterator n=c->nodes.begin();
            n != c->nodes.end();
            n++) {
                sumDeg += n-> degree();
        }
        mod -= sumDeg * sumDeg / 2. / nEdges;
    }

    // NOTE: this is useless for the optimization, but oh so cheap
    mod /= 2 * nEdges;

    return mod;
}


//FIXME: this is totally unclear, look it up
std::vector<uint64_t> Network::nodesInRadomOrder(uint32_t seed) {
    using RandomSource = boost::mt19937;    
    RandomSource randomSource(seed);

    std::vector<uint64_t> randomOrder(nodes.size());
    boost::random::shuffle_order_engine<
    boost::random::linear_congruential_engine<uint32_t, 1366, 150889, 714025>,
    97> kreutzer1986();

    //kreutzer1986.generate(randomOrder.begin(), randomOrder.end());




    return randomOrder;
}


// When you flip a node, figure what cluster flip increases the modularity most
// The algorithm has two parts:
// 1. figure out the list of clusters this node has edges with (neighboring clusters)
// 2. go over the neighboring clusters and calculate the difference in modularity
uint64_t Network::findBestCluster(uint64_t nodeId) {

    // 1. Make list of neighboring clusters
    uint64_t origClusterId;
    std::vector<uint64_t> neighborsId;
    std::set<uint64_t> neighboringClusters;
    for(std::vector<Node>::iterator n=nodes.begin();
        n != nodes.end();
        n++) {
        if(n->nodeId == nodeId) {
            neighborsId = n->neighbors;
            origClusterId = n->cluster;
            // We want to keep the original cluster as an option, to not force change
            neighboringClusters.insert(n->cluster);
            break;
        }
    }
    for(std::vector<Cluster>::iterator c=clusters.begin();
        c != clusters.end();
        c++) {
        for(std::vector<Node>::iterator n=c->nodes.begin();
            n != c->nodes.end();
            n++) {
            if(std::find(neighborsId.begin(), neighborsId.end(), n->nodeId) != neighborsId.end()) {
                neighboringClusters.insert(c->clusterId);
            }
        } 
    }

    // 2. Compute the best cluster
    double mod;
    // This is ok because not moving the node has delta mod = 0 > -1
    double modMax = -1;
    uint64_t clusterIdMax;
    for(auto c=clusters.begin();
        c != clusters.end();
        c++) {
        //if(std::find(neighboringClusters.begin(), neighboringClusters.end(), c->clusterId) == neighboringClusters.end())
        //    continue;

        // If the node stays where it is, the diff of modularity is zero
        // this is always possible (stable node)
        if(c->clusterId == origClusterId) {
            mod = 0;
            if(mod > modMax) {
                modMax = mod;
                clusterIdMax = c->clusterId;
            }
        } else {
            // The cluster gains a few internal links (with nodeId)
            // The original cluster might lose a few links
            // vice versa with the squared sums of degrees
            // so there are 4 terms in this evaluation
            mod = 0;
            for(std::vector<Node>::iterator n=nodes.begin();
                n != nodes.end();
                n++) {
                if(n->nodeId == nodeId) {
                    // Adding this node to the cluster can add internal edges
                    // check all nodes in this cluster for edges
                    for(std::vector<Node>::iterator n2=c->nodes.begin();
                        n2 != c->nodes.end();
                        n2++) {
                        if(std::find(neighborsId.begin(), neighborsId.end(), n2->nodeId) != neighborsId.end()) {
                            mod += 1.0 / (2 * nEdges);
                        }
                    }
                    // Removing the node from the original cluster can remove edges
                    for(std::vector<Cluster>::iterator c2=clusters.begin();
                        c2 != clusters.end();
                        c2++) {
                        if(c2->clusterId == origClusterId) {
                            for(std::vector<Node>::iterator n2=c2->nodes.begin();
                                n2 != c2->nodes.end();
                                n2++) {
                                if(std::find(neighborsId.begin(), neighborsId.end(), n2->nodeId) != neighborsId.end()) {
                                    mod -= 1.0 / (2 * nEdges);
                                }
                            }
                        break;
                        }
                    }
                    break;
                }
            }
            if(mod > modMax) {
                modMax = mod;
                clusterIdMax = c->clusterId;
            }
        }
    }


    return 4;
}


void Network::updateCluster(uint64_t nodeId, uint64_t clusterId) {

    // Update the node list
    uint64_t clusterIdOld;
    Node newNode(nodeId, clusterId);
    for(std::vector<Node>::iterator n=nodes.begin();
        n != nodes.end();
        n++) {
        if(n->nodeId == nodeId) {
            clusterIdOld = n->cluster;
            n->cluster = clusterId;
            newNode.neighbors = n->neighbors;
            break;
        }
    }

    // Update the cluster list
    bool clusterNewFound = false;
    bool clusterEmptyFound = false;
    std::vector<Cluster>::iterator clusterEmpty;
    for(std::vector<Cluster>::iterator c=clusters.begin();
        c != clusters.end();
        c++) {
        if(c->clusterId == clusterId) {
            c->nodes.push_back(newNode);
            clusterNewFound = true;
        } else if(c->clusterId == clusterIdOld) {
            for(std::vector<Node>::iterator n=c->nodes.begin();
                n != c->nodes.end();
                n++) {
                if(n->nodeId == nodeId) {
                    c->nodes.erase(n);
                    if(c->nodes.size() == 0) {
                            clusterEmpty = c;
                            clusterEmptyFound = true;
                    }
                    break;
                }
            }
        }
    }
    if(clusterEmptyFound)
        clusters.erase(clusterEmpty);
    if(!clusterNewFound) {
        std::vector<Node> nodesNewCluster({newNode});
        Cluster newCluster(clusterId, nodesNewCluster);
        clusters.push_back(newCluster);
    }

}

bool Network::runLocalMovingAlgorithm(uint32_t randomSeed) {
    double mod = 0;

    if(nNodes == 1)
        return false;

    bool update = false;

    // Add randomization ;-)
    std::vector<uint64_t> nodesShuffled = nodesInRadomOrder(randomSeed);

    uint64_t numberStableNodes = 0;
    int i = 0;
    uint64_t nodeId;
    uint64_t bestClusterId;
    bool isStable;
    do {
        nodeId = nodesShuffled[i];
        isStable = false;

        // Find best cluster for the random node, including its own one
        bestClusterId = findBestCluster(nodeId);

        // If the best cluster was already set, the node is stable
        for(std::vector<Node>::iterator n=nodes.begin();
            n != nodes.end();
            n++) {
            if(n->nodeId == nodeId) {
                if(n->cluster == bestClusterId) {
                    numberStableNodes++;
                    isStable = true;
                }
                break;
            }
        }
        if(!isStable) {
            updateCluster(nodeId, bestClusterId); 
            numberStableNodes = 1;
            update = true;
        }

        i = (i < nodes.size() - 1) ? (i + 1) : 0;
    
    } while(numberStableNodes < nNodes);

    return update;
}
