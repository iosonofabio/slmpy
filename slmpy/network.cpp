#include <iostream>
#include <set>
#include <map>
#include <cmath>
#include <algorithm> // std::random_shuffle
#include <cstdlib>   // std::rand, std::srand

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <slmpy.h>


void Network::fromPython(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edgesIn,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > clustersIn) {

    // Set number of nodes and edges
    nNodes = nodesIn.rows();
    nEdges = edgesIn.rows();

    // Count the clusters
    std::set<uint64_t> clusterIdSet;
    for(size_t i=0; i < clustersIn.rows(); i++) {
        clusterIdSet.insert(clustersIn(i, 0));
    }

    // Make empty communities
    for(auto cid = clusterIdSet.begin(); cid != clusterIdSet.end(); cid++) {
        Cluster c(*cid);
        clusters.push_back(c);
    }

    // Fill the nodes
    std::cout << "Fillig the nodes from Python: ";
    for(uint64_t i=0; i < nNodes; i++) {
        Node n(nodesIn(i, 0), clustersIn(i, 0));
        nodes.push_back(n);
        std::cout << n.nodeId << " ";
    }
    std::cout << std::endl << std::flush;

    // Fill the edges (both directions)
    // NOTE: this assumes that the input edges are unique, not redundant
    uint64_t tmp;
    uint64_t tmp2;
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


// Fill output vector
void Network::toPython(
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communities_out) {

    std::vector<uint64_t> communities_from_network = getClusterIds();

    uint64_t irow = 0;
    for(std::vector<long unsigned int>::iterator c=communities_from_network.begin();
        c != communities_from_network.end();
        c++) {
        communities_out(irow++, 0) = *c;
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


// shuffle order of nodes that get probes by the local moving heuristic
std::vector<uint64_t> Network::nodesInRadomOrder(uint32_t seed) {
    std::srand(seed);
    std::vector<uint64_t> randomOrder(nNodes);

    std::cout << "Randomizing order, original order: ";
    for(size_t i=0; i != nodes.size(); i++) {
        randomOrder[i] = nodes[i].nodeId;
        std::cout << randomOrder[i] << " ";
    }
    std::cout << std::endl << std::flush;

    std::random_shuffle(randomOrder.begin(), randomOrder.end());
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
        if(std::find(neighboringClusters.begin(), neighboringClusters.end(), c->clusterId) == neighboringClusters.end())
            continue;

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
                    // Adding node to the new cluster
                    for(std::vector<Node>::iterator n2=c->nodes.begin();
                        n2 != c->nodes.end();
                        n2++) {
                        // 1. This can add internal edges
                        if(std::find(neighborsId.begin(), neighborsId.end(), n2->nodeId) != neighborsId.end()) {
                            mod += 1.0 / (2 * nEdges);
                        }
                        // 2. Subtract k_i sum_j k_j / (2m)^2 from the new cluster
                        mod -= 1.0 * n->degree() * n2->degree() / (2 * nEdges) / (2 * nEdges);
                    }

                    // Removing the node from the original cluster
                    for(std::vector<Cluster>::iterator c2=clusters.begin();
                        c2 != clusters.end();
                        c2++) {
                        if(c2->clusterId == origClusterId) {
                            for(std::vector<Node>::iterator n2=c2->nodes.begin();
                                n2 != c2->nodes.end();
                                n2++) {
                                // 3. This can remove edges
                                if(std::find(neighborsId.begin(), neighborsId.end(), n2->nodeId) != neighborsId.end()) {
                                    mod -= 1.0 / (2 * nEdges);
                                }
                                // 4. Add k_i sum_j k_j / (2m)^2 to the old cluster
                                mod += 1.0 * n->degree() * n2->degree() / (2 * nEdges) / (2 * nEdges);
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
    return clusterIdMax;
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

bool Network::runLocalMovingAlgorithm(uint32_t randomSeed, int64_t maxIterations) {
    bool update = false;

    if(nNodes == 1)
        return update;

    std::cout << "runLocalMovingAlgorithm, shuffling nodes" << std::endl << std::flush;
    std::vector<uint64_t> nodesShuffled = nodesInRadomOrder(randomSeed);
    for(size_t i=0; i!=nNodes; i++)
        std::cout << nodesShuffled[i] << " ";
    std::cout << std::endl << std::flush;

    uint64_t numberStableNodes = 0;
    int i = 0;
    uint64_t nodeId;
    uint64_t bestClusterId;
    bool isStable;
    int64_t iteration = 0;
    do {
        std::cout << "runLocalMovingAlgorithm, iteration " << (iteration + 1) << std::endl << std::flush;

        isStable = false;
        nodeId = nodesShuffled[i];
        std::cout << "random node id: " << nodeId << std::endl << std::flush;

        std::cout << "findBestCluster" << std::endl << std::flush;
        // Find best cluster for the random node, including its own one
        bestClusterId = findBestCluster(nodeId);
        std::cout << "bestClusterId: " << bestClusterId << std::endl << std::flush;

        std::cout << "check for stability" << std::endl << std::flush;
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

        std::cout << "stable: " << isStable << std::endl << std::flush;

        if(!isStable) {
            std::cout << "node is unstable, updateCluster" << std::endl << std::flush;

            updateCluster(nodeId, bestClusterId); 
            numberStableNodes = 1;
            update = true;

            std::cout << "clusters updated" << std::endl << std::flush;
        }

        // cycle around the random vector
        i = (i < nodesShuffled.size() - 1) ? (i + 1) : 0;

        std::cout << "end of iteration, looping with i = " << i << std::endl << std::endl << std::flush;

        iteration++;
        if(iteration == maxIterations)
            break;
    
    } while(numberStableNodes < nNodes);

    return update;
}


std::vector<uint64_t> Network::getClusterIds() {
    std::vector<uint64_t> clusterIds(nodes.size());
    for(size_t i=0; i<nodes.size(); i++) {
        clusterIds[i] = nodes[i].cluster; 
    }
    return clusterIds;
}
