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
    uint64_t nEdges = edgesIn.rows();

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
    for(uint64_t i=0; i < nNodes; i++) {
        Node n(nodesIn(i, 0), clustersIn(i, 0));
        nodes[n.nodeId] = n;
    }

    // Fill the edges (both directions)
    // NOTE: this assumes that the input edges are unique, not redundant
    uint64_t tmp;
    uint64_t tmp2;
    for(uint64_t edgeId=0; edgeId < nEdges; edgeId++) {
        tmp = edgesIn(edgeId, 0);
        tmp2 = edgesIn(edgeId, 1);
        // Unweighted edges get a weight of 1
        nodes[tmp].neighbors[tmp2] = 1;
        nodes[tmp2].neighbors[tmp] = 1;
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

    // rename communities as 0-N, by inverse size
    std::vector<std::pair<int64_t, uint64_t>> clusterIds;
    size_t i = 0;
    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        // std::sort uses the first element, ascending
        std::pair<int64_t, uint64_t> cl(-(c->nodes.size()), c->clusterId);
        clusterIds.push_back(cl);
        //std::cout << "Cluster: " << c->clusterId << " size: " << c->nodes.size() << std::endl << std::flush;
    }
    std::sort(clusterIds.begin(), clusterIds.end());
    std::map<uint64_t, uint64_t> clusterRename;
    for(size_t i=0; i != clusters.size(); i++) {
        clusterRename[clusterIds[i].second] = i;
    }

    for(size_t i=0; i<nodes.size(); i++) {
        communitiesOut(i, 0) = clusterRename[nodes[nodesIn(i, 0)].cluster];
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

    double nEdges2 = calcTwiceTotalEdges(); 

    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        // Calculate twice the number of edges within the community
        for(auto n=c->nodes.begin(); n != c->nodes.end(); n++) {
            Node& node = nodes[*n];
            for(auto neiId=node.neighbors.begin(); neiId != node.neighbors.end(); neiId++) {
                for(auto n2=c->nodes.begin(); n2 != c->nodes.end(); n2++) {
                    if((*n2) == (neiId->first)) {
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
        mod -= sumDeg * sumDeg / nEdges2;
    }

    // NOTE: this is useless for the optimization, but oh so cheap
    mod /= nEdges2;

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


// calculate the total edge weight of the network
double Network::calcTwiceTotalEdges() {
    double w = 0;
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        for(auto nei = n->second.neighbors.begin(); nei != n->second.neighbors.end(); nei++) {
            w += nei->second;
        }
    }
    return w;
}

// When you flip a node, figure what cluster flip increases the modularity most
// The algorithm has two parts:
// 1. figure out the list of clusters this node has edges with (neighboring clusters)
// 2. go over the neighboring clusters and calculate the difference in modularity
uint64_t Network::findBestCluster(uint64_t nodeId) {

    // 1. Make list of neighboring clusters
    Node& node = nodes[nodeId];
    uint64_t origClusterId = node.cluster;
    std::vector<Cluster>::iterator origCluster;
    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        if(c->clusterId == origClusterId) {
            origCluster = c;
        }
    }
    double nEdges2 = calcTwiceTotalEdges();

    // Make a list of potential clusters, including the current one
    std::set<uint64_t> neighboringClusters({node.cluster});
    for(auto n=node.neighbors.begin(); n != node.neighbors.end(); n++) {
        neighboringClusters.insert(nodes[n->first].cluster);
    }

    // 2. Compute the best cluster
    // NOTE: all costs are calculated * nEdges2, as it makes no difference for the argmax
    // If the node stays where it is, the diff of modularity is zero
    // this is always possible (stable node)
    double modMax = 0;
    uint64_t clusterIdMax = node.cluster;

    // Calculate the cost of leaving your cluster once for all
    double modLeaving = 0;
    // lost edges (you lose the self-edge)
    for(auto nei=node.neighbors.begin(); nei != node.neighbors.end(); nei++) {
        if (nodes[nei->first].cluster == origClusterId) {
            modLeaving -= nei->second;         
        }
    }
    // baseline degrees
    double nodeDeg = node.degree();
    for(auto ni2=origCluster->nodes.begin(); ni2 != origCluster->nodes.end(); ni2++) {
        modLeaving += nodeDeg * nodes[*ni2].degree() / nEdges2;
    }

    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        if(std::find(neighboringClusters.begin(), neighboringClusters.end(), c->clusterId) == neighboringClusters.end())
            continue;
        if(c->clusterId == origClusterId)
            continue;

        // Leaving your cluster has a fixed difference in modularity
        double mod = modLeaving;

        // The cluster gains a few internal links (with nodeId)
        // The original cluster might lose a few links
        // vice versa with the squared sums of degrees
        // so there are 4 terms in this evaluation

        // Check additional edges into this cluster
        // the self-edge comes back
        for(auto nei=node.neighbors.begin(); nei != node.neighbors.end(); nei++) {
            if((nodes[nei->first].cluster == c->clusterId) || (nei->first == nodeId)){
                mod += nei->second;         
            }
        }
        // Subtract k_i sum_j k_j / (2m)^2 from the new cluster
        for(auto ni2=c->nodes.begin(); ni2 != c->nodes.end(); ni2++) {
            mod -= nodeDeg * nodes[*ni2].degree() / nEdges2;
        }
        // Subtract your own weight, since you belong to the new cluster now
        mod -= nodeDeg * nodeDeg / nEdges2;

        if(mod > modMax) {
            modMax = mod;
            clusterIdMax = c->clusterId;
        }
    }
    return clusterIdMax;
}


void Network::updateCluster(uint64_t nodeId, uint64_t clusterId) {

#if SLMPY_VERBOSE
    std::cout << "updateCluster " << std::endl << std::flush;
    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        std::cout << "Cluster: " << c->clusterId << " size: " << c->nodes.size() << std::endl << std::flush;
    }
#endif


    // Update the node list
    Node& node = nodes[nodeId];
    uint64_t clusterIdOld = node.cluster;
    node.cluster = clusterId;

    // Update the cluster list
    bool clusterNewFound = false;
    int bothFound = 0;
    std::vector<Cluster>::iterator clusterEmpty;
    for(auto c=clusters.begin(); c != clusters.end(); c++) {

        // remove node from old cluster
        if(c->clusterId == clusterIdOld) {
            for(auto n=c->nodes.begin(); n != c->nodes.end(); n++) {
                if((*n) == nodeId) {
                    if(c->nodes.size() == 1) {
                        // erase shifts to the new element, but there is c++
                        c = clusters.erase(c) - 1;
                    } else {
                        c->nodes.erase(n);
                    }
                    bothFound++;
                    break;
                }
            }
        // add node to new cluster
        } else if(c->clusterId == clusterId) {
            clusterNewFound = true;
            bothFound++;
            c->nodes.push_back(nodeId);
        }
        if(bothFound == 2)
            break;
    }
    // if the clusterId is not present, put it into a new one
    if(!clusterNewFound) {
        std::vector<uint64_t> nodesNewCluster({nodeId});
        Cluster newCluster(clusterId, nodesNewCluster);
        clusters.push_back(newCluster);
    }

#if SLMPY_VERBOSE
    std::cout << "post updateCluster " << std::endl << std::flush;
    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        std::cout << "Cluster: " << c->clusterId << " size: " << c->nodes.size() << std::endl << std::flush;
    }
#endif

}

bool Network::runLocalMovingAlgorithm(uint32_t randomSeed) {
    bool update = false;
    if(nNodes == 1)
        return update;

    std::vector<uint64_t> nodesShuffled = nodesInRadomOrder(randomSeed);

    uint64_t numberStableNodes = 0;
    int i = 0;
    uint64_t nodeId;
    uint64_t bestClusterId;
    bool isStable;
    int64_t iteration = 0;
    do {
#if SLMPY_VERBOSE
        std::cout << "runLocalMovingAlgorithm, iteration " << (iteration + 1) << std::endl << std::flush;
#endif

        isStable = false;
        nodeId = nodesShuffled[i];

#if SLMPY_VERBOSE
        std::cout << "findBestCluster" << std::endl << std::flush;
#endif
        // Find best cluster for the random node, including its own one
        bestClusterId = findBestCluster(nodeId);
#if SLMPY_VERBOSE
        std::cout << "bestClusterId: " << bestClusterId << std::endl << std::flush;
#endif

        //FIXME
        if(nodes.size() < 10)
            std::cout << "bestCluster: node " << nodeId << ", old cluster " << nodes[nodeId].cluster << ", new cluster " << bestClusterId << std::endl << std::flush;

        // If the best cluster was already set, the node is stable
#if SLMPY_VERBOSE
        std::cout << "check for stability" << std::endl << std::flush;
#endif
        if(nodes[nodeId].cluster == bestClusterId) {
            numberStableNodes++;
            isStable = true;
        }
#if SLMPY_VERBOSE
        std::cout << "stable: " << isStable << std::endl << std::flush;
#endif

        if(!isStable) {
#if SLMPY_VERBOSE
            std::cout << "node is unstable, updateCluster" << std::endl << std::flush;
#endif

            updateCluster(nodeId, bestClusterId); 
            numberStableNodes = 1;
            update = true;

#if SLMPY_VERBOSE
            std::cout << "clusters updated" << std::endl << std::flush;
#endif
        }

        // cycle around the random vector
        i = (i < nodesShuffled.size() - 1) ? (i + 1) : 0;

#if SLMPY_VERBOSE
        std::cout << "end of iteration, looping with i = " << i << std::endl << std::endl << std::flush;
#endif

        iteration++;
    
    } while(numberStableNodes < nNodes);

    return update;
}


bool Network::runLouvainAlgorithm(uint32_t randomSeed) {
    bool update = false;
    bool update2;
    if(nNodes == 1)
        return update;

    update |= runLocalMovingAlgorithm(randomSeed);

    if(clusters.size() == nodes.size())
        return update;

    Network redNet = calculateReducedNetwork();
    redNet.createSingletons();
    update2 = redNet.runLouvainAlgorithm(randomSeed);

    //check the reduced network
    std::cout<<"Reduced after LM:"<<std::endl<<std::flush;
    for(auto n=redNet.nodes.begin(); n != redNet.nodes.end(); n++) {
        std::cout << n->first << ", weights: ";
       for(auto nei=n->second.neighbors.begin(); nei != n->second.neighbors.end(); nei++) {
           if(nei->first == n->first)
                continue;
           std::cout << "(" << nei->first << ", " << nei->second << ") ";
       }
       std::cout << std::endl << std::flush;
    }
    std::cout << "Updated: " << update2 << std::endl << std::flush;
    std::cout << std::endl << std::flush;

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

    update |= runLocalMovingAlgorithm(randomSeed);

    if(clusters.size() == nodes.size())
        return update;
    
    // each community -> subnetwork, reset labels and runLocalMoving inside that subnetwork
    // then set the cluster ids as of the double splitting
    std::vector<Network> subnetworks = createSubnetworks();
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
        nClusters += subNet.clusters.size();
    }

    Network redNet = calculateReducedNetwork();
    // the initial state is not each reduced node for itself, but rather
    // each subnetwork for itself. This is probably for convergence/speed reasons
    redNet.createFromSubnetworks(subnetworks);

    update |= redNet.runSmartLocalMovingAlgorithm(randomSeed, maxIterations);
    mergeClusters(redNet.clusters);

    return update;
}


// Initialize network by putting each node in its own community
void Network::createSingletons() {
    uint64_t clusterId = 0;
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        (n->second).cluster = clusterId++;
    }
    calcClustersFromNodes();
}


// the next two functions go up and down the reduced network recursion
Network Network::calculateReducedNetwork() {

    std::cout<<"Reducing network"<<std::endl<<std::flush;

    Network redNet;
    redNet.nNodes = clusters.size();

    // Make map of clusterId to reduced network ordering
    std::map<uint64_t, uint64_t> clusterMap;
    uint64_t i = 0;
    for(auto c=clusters.begin(); c != clusters.end(); c++, i++) {
        clusterMap[c->clusterId] = i;
    }

    // set edge weights
    // in the original implementation, every reduced networks is a complete graph
    // with possible zero weights. Here we are a bit more lenient
    i = 0;
    for(auto c=clusters.begin(); c != clusters.end(); c++, i++) {
        // nodes in the reduced network are numbered 0-x
        Node n(i);

        // set edge weights out of this reduced node
        // self weight emerges naturally here, because some neighbors of the
        // parent node will be in the same cluster and that edge is added to
        // weights[i] which is the self weight
        std::vector<double> weights(clusters.size(), 0);
        for(auto nid=c->nodes.begin(); nid != c->nodes.end(); nid++) {
            for(auto nn = nodes[*nid].neighbors.begin(); nn != nodes[*nid].neighbors.end(); nn++) {
                weights[clusterMap[nodes[nn->first].cluster]] += nn->second;
            }
        }

        // set nonzero neighbors
        // in the reduced network, a node can be its own neighbor (self-links)
        // that happens if the parent network had self links or, in the first
        // reduction, if there is at least one link inside the community
        uint64_t j = 0;
        for(auto w=weights.begin(); w != weights.end(); w++, j++) {
            if((*w) > 0) {
                n.neighbors[j] = *w;
            }
        }

        redNet.nodes[i] = n;
    }

    //check the reduced network
    std::cout<<"Reduced:"<<std::endl<<std::flush;
    for(auto n=redNet.nodes.begin(); n != redNet.nodes.end(); n++) {
        std::cout << n->first << ", weights: ";
       for(auto nei=n->second.neighbors.begin(); nei != n->second.neighbors.end(); nei++) {
           if(nei->first == n->first)
                continue;
           std::cout << "(" << nei->first << ", " << nei->second << ") ";
       }
       std::cout << std::endl << std::flush;
    }
    std::cout << std::endl << std::flush;


    return redNet;
}


void Network::calcClustersFromNodes() {
    clusters.clear();
    std::set<uint64_t> clusterIdSet;
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        uint64_t cId = n->second.cluster;
        if(clusterIdSet.find(cId) == clusterIdSet.end()) {
            Cluster newCluster(cId);
            newCluster.nodes.push_back(n->first);
            clusters.push_back(newCluster);
            clusterIdSet.insert(cId);
        } else {
            for(auto c=clusters.begin(); c != clusters.end(); c++) {
                if(c->clusterId == cId) {
                    c->nodes.push_back(n->first);
                    break;
                }
            }
        }
    }

}

// propagate the clusters from a reduced network up the chain
void Network::mergeClusters(std::vector<Cluster> clustersRed) {

    // iterate over the merged clusters
    uint64_t nClu = 0;
    for(auto c=clustersRed.begin(); c != clustersRed.end(); c++, nClu++) {
        // all nodes in here belong to the same community
        // however, each of c->nodes is a whole set of nodes in the parent network
        // the c->nodes[X].cluster corresponds to the X-th element in the parent
        // cluster list, that's the way it was set up when reducing
        for(auto cId=c->nodes.begin(); cId != c->nodes.end(); cId++) {
            for(auto nId=clusters[*cId].nodes.begin(); nId != clusters[*cId].nodes.end(); nId++) {
                nodes[*nId].cluster = nClu;
            }
        }
    }
    
    // regenerate clusters from the nodes
    calcClustersFromNodes();

    std::cout << "Clusters after merging: " << std::endl << std::flush;
    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        std::cout << "Cluster: " << c->clusterId << " size: " << c->nodes.size() << std::endl << std::flush;
    }
}


// the next two functions go up and down the subnetwork business
void Network::createFromSubnetworks(std::vector<Network> subnetworks) {
    // TODO
    1;
}


std::vector<Network> Network::createSubnetworks() {
    std::vector<Network> subnetworks;
    // TODO: implement
    return subnetworks;
}
