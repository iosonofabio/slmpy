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
        nodes[n.nodeId] = n;
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
        nodes[tmp].neighbors.push_back(tmp2);
        nodes[tmp2].neighbors.push_back(tmp);
    }

    // Fill the clusters
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        for(auto c=clusters.begin(); c != clusters.end(); c++) {
            if((n->second).cluster == c->clusterId) {
                c->nodes.push_back(n->first);
                break;
            }
        }
    }
}


// Fill output vector
void Network::toPython(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communitiesOut) {

    for(size_t i=0; i<nodes.size(); i++) {
        communitiesOut(i, 0) = nodes[nodesIn(i, 0)].cluster;
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

    for(auto c=clusters.begin(); c != clusters.end(); c++) {

        // Calculate twice the number of edges within the community
        for(auto n=c->nodes.begin(); n != c->nodes.end(); n++) {
            Node& node = nodes[*n];
            for(auto neiId=node.neighbors.begin(); neiId != node.neighbors.end(); neiId++) {
                for(auto n2=c->nodes.begin(); n2 != c->nodes.end(); n2++) {
                    if((*n2) == (*neiId)) {
                        mod += 1;
                        break;
                    }
                }
            }
        }

        // Subtract the summed degrees
        double sumDeg = 0;
        for(auto n=c->nodes.begin(); n != c->nodes.end(); n++) {
                sumDeg += nodes[*n].degree();
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

    size_t i = 0;
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        randomOrder[i++] = (n->second).nodeId;
    }

    std::random_shuffle(randomOrder.begin(), randomOrder.end());
    return randomOrder;
}


// When you flip a node, figure what cluster flip increases the modularity most
// The algorithm has two parts:
// 1. figure out the list of clusters this node has edges with (neighboring clusters)
// 2. go over the neighboring clusters and calculate the difference in modularity
uint64_t Network::findBestCluster(uint64_t nodeId) {

    // 1. Make list of neighboring clusters
    Node node = nodes[nodeId];
    uint64_t origClusterId = node.cluster;
    std::vector<uint64_t> neighborsId = node.neighbors;
    // We want to keep the original cluster as an option, to not force change
    std::set<uint64_t> neighboringClusters({node.cluster});
    for(auto n=neighborsId.begin(); n != neighborsId.end(); n++) {
        neighboringClusters.insert(nodes[*n].cluster);
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
            Node& n = nodes[nodeId];
            Node n2;

            // Hypothetical adding node to the new cluster
            for(auto ni2=c->nodes.begin(); ni2 != c->nodes.end(); ni2++) {
                n2 = nodes[*ni2];

                // 1. This can add internal edges
                if(std::find(neighborsId.begin(), neighborsId.end(), n2.nodeId) != neighborsId.end()) {
                    mod += 1.0 / (2 * nEdges);
                }
                // 2. Subtract k_i sum_j k_j / (2m)^2 from the new cluster
                mod -= 1.0 * n.degree() * n2.degree() / (2 * nEdges) / (2 * nEdges);
            }

            // Hypothetical removing the node from the original cluster
            for(auto c2=clusters.begin(); c2 != clusters.end(); c2++) {
                if(c2->clusterId == origClusterId) {
                    for(auto ni2=c2->nodes.begin(); ni2 != c2->nodes.end(); ni2++) {
                        n2 = nodes[*ni2];
                            
                        // 3. This can remove edges
                        if(std::find(neighborsId.begin(), neighborsId.end(), n2.nodeId) != neighborsId.end()) {
                            mod -= 1.0 / (2 * nEdges);
                        }
                        // 4. Add k_i sum_j k_j / (2m)^2 to the old cluster
                        mod += 1.0 * n.degree() * n2.degree() / (2 * nEdges) / (2 * nEdges);
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
    Node& node = nodes[nodeId];
    uint64_t clusterIdOld = node.cluster;
    node.cluster = clusterId;

    // Update the cluster list
    bool clusterNewFound = false;
    bool clusterEmptyFound = false;
    std::vector<Cluster>::iterator clusterEmpty;
    for(auto c=clusters.begin();
        c != clusters.end();
        c++) {
        if(c->clusterId == clusterId) {
            c->nodes.push_back(nodeId);
            clusterNewFound = true;
        } else if(c->clusterId == clusterIdOld) {
            for(auto n=c->nodes.begin();
                n != c->nodes.end();
                n++) {
                if((*n) == nodeId) {
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
        std::vector<uint64_t> nodesNewCluster({nodeId});
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

        // If the best cluster was already set, the node is stable
        std::cout << "check for stability" << std::endl << std::flush;
        if(nodes[nodeId].cluster == bestClusterId) {
            numberStableNodes++;
            isStable = true;
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


bool Network::runLouvain(uint32_t randomSeed) {
    bool update = false;
    bool update2;
    if(nNodes == 1)
        return update;

    update |= runLocalMovingAlgorithm(randomSeed, 3 * nNodes);

    if(clusters.size() == nodes.size())
        return update;

    ReducedNetwork redNet = calculateReducedNetwork();
    redNet.createSingletons();
    update2 = redNet.runLouvainAlgorithm(randomSeed);
    if(update2) {
        update = true;
        mergeClusters(redNet.clusters);
    }
    return update;
}


bool Network::runSmartLocalMovingAlgorithm(uint32_t randomSeed, int64_t maxIterations) {
    bool update = false;
    if(nNodes == 1)
        return update;

    update |= runLocalMovingAlgorithm(randomSeed, 3 * nNodes);

    if(clusters.size() == nodes.size())
        return update;
    
    // each community -> subnetwork, reset labels and runLocalMoving inside that subnetwork
    // then set the cluster ids as of the double splitting
    std::vector<Network> subnetworks;
    subnetworks.create();
    uint64_t nClusters = 0;
    for(size_t isn=0; isn != subnetworks.size(); isn++) {
        Network& subNet = subnetworks[isn];
        // reset labels
        subNet.createSingletons();
        // cluster within subnetwork
        subNet.runLocalMovingAlgorithm(randomSeed);

        // assign community numbers to all nodes across the parent network
        for(auto n=subNet.nodes.begin(); n!=subNet.nodes.end(); n++) {
            // n->cluster has the clusterId within the subnetwork, go to the
            // global list of nodes and update it with a new clusterId that
            // runs over all subnetworks
            nodes[n->first].cluster = nClusters + n->second.cluster;
        }
        nClusters += subnetwork.clusters.size();
    }

    ReducedNetwork redNet = calculateReducedNetwork();
    // the initial state is not each reduced node for itself, but rather
    // each subnetwork for itself. This is probably for convergence/speed reasons
    redNet.createFromSubnetworks(subnetworks);

    update |= redNet.runSmartLocalMovingAlgorithm(randomSeed, maxIterations);
    mergeClusters(redNet.clusters);

    return update;
}


