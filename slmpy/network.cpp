#include <iostream>
#include <set>
#include <map>
#include <cmath>
#include <algorithm> // std::random_shuffle
#include <cstdlib>   // std::rand, std::srand
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <slmpy.h>


void Network::fromPython(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 2> > edgesIn,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > clustersIn,
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > fixedNodesIn) {

    // Fill the fixed nodes
    for(int i=0; i < fixedNodesIn.rows(); i++) {
        fixedNodes.insert(fixedNodesIn(i, 0));
    }

    // Fill the nodes
    for(int i=0; i < nodesIn.rows(); i++) {
        Node n(clustersIn(i, 0));
        nodes[nodesIn(i, 0)] = n;
    }

    // Count the clusters
    std::set<uint64_t> clusterIdSet;
    for(int i=0; i < clustersIn.rows(); i++) {
        clusterIdSet.insert(clustersIn(i, 0));
    }

    // Make empty communities
    for(auto cid = clusterIdSet.begin(); cid != clusterIdSet.end(); cid++) {
        Cluster c(*cid);
        clusters.push_back(c);
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

    // Fill the edges (both directions)
    // NOTE: this assumes that the input edges are unique, not redundant
    uint64_t tmp;
    uint64_t tmp2;
    for(int edgeId=0; edgeId < edgesIn.rows(); edgeId++) {
        tmp = edgesIn(edgeId, 0);
        tmp2 = edgesIn(edgeId, 1);
        // Unweighted edges get a weight of 1
        nodes[tmp].neighbors[tmp2] = 1;
        nodes[tmp2].neighbors[tmp] = 1;
    }

    calcDegreesAndTwiceTotalEdges();
}


// Fill output vector
void Network::toPython(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, 1> > nodesIn,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, 1> > communitiesOut) {

    // rename communities as 0-N, by inverse size
    std::vector<std::pair<int64_t, uint64_t>> clusterIds;
    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        // std::sort uses the first element, ascending
        std::pair<int64_t, uint64_t> cl(-(c->nodes.size()), c->clusterId);
        clusterIds.push_back(cl);
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


// shuffle order of nodes that get probes by the local moving heuristic
std::vector<uint64_t> Network::nodesInRadomOrder() {
    std::vector<uint64_t> randomOrder(nodes.size());

    size_t i = 0;
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        randomOrder[i++] = n->first;
    }

    // FIXME: we seem to have issues here!!
    std::shuffle(randomOrder.begin(), randomOrder.end(), urng);
    //std::random_shuffle(randomOrder.begin(), randomOrder.end());

    return randomOrder;
}


// calculate the total edge weight of the network
void Network::calcDegreesAndTwiceTotalEdges() {
    twiceTotalEdges = 0;
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        double w = 0;
        for(auto nei = n->second.neighbors.begin(); nei != n->second.neighbors.end(); nei++) {
            w += nei->second;
        }
        n->second.degree = w;
        twiceTotalEdges += w;
    }
}


// Initialize network by putting each node in its own community
void Network::createSingletons() {
    uint64_t clusterId = 0;
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        (n->second).cluster = clusterId++;
    }
    calcClustersFromNodes();
    calcDegreesAndTwiceTotalEdges();
}


// the next two functions go up and down the reduced network recursion
Network Network::calculateReducedNetwork() {

#if SLMPY_VERBOSE
    std::cout<<"Reducing network"<<std::endl<<std::flush;
#endif

    Network redNet;

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
        // nodes in the reduced network are initilized to their clusterId
        Node n(c->clusterId);

        // set edge weights out of this reduced node
        // self weight emerges naturally here, because some neighbors of the
        // parent node will be in the same cluster and that edge is added to
        // weights[i] which is the self weight
        std::vector<double> weights(clusters.size(), 0);
        for(auto nid=c->nodes.begin(); nid != c->nodes.end(); nid++) {
            // if a cluster contains a fixed node, the whole cluster is fixed
            // in the reduced network, else it could merge
            if(fixedNodes.find(*nid) != fixedNodes.end()) {
                redNet.fixedNodes.insert(i);
            }
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

        // nodes in the reduced network are numbered 0-x
        redNet.nodes[i] = n;
    }

#if SLMPY_VERBOSE
    check the reduced network
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
#endif


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

#if SLMPY_VERBOSE
    std::cout << "Clusters after merging: " << std::endl << std::flush;
    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        std::cout << "Cluster: " << c->clusterId << " size: " << c->nodes.size() << std::endl << std::flush;
    }
#endif
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

    // Make a list of potential clusters, including the current one
    std::set<uint64_t> neighboringClusters({node.cluster});
    for(auto n=node.neighbors.begin(); n != node.neighbors.end(); n++) {
        neighboringClusters.insert(nodes[n->first].cluster);
    }

    // 2. Compute the best cluster
    // NOTE: all costs are calculated * 2m, as it makes no difference for the argmax
    // If the node stays where it is, the diff of modularity is zero
    // this is always possible (stable node)
    double modMax = 0;
    uint64_t clusterIdMax = node.cluster;

    // Calculate the cost of leaving your cluster once for all
    double modLeaving = 0;
    // lost edges (excluding the self-edge)
    for(auto nei=node.neighbors.begin(); nei != node.neighbors.end(); nei++) {
        if(nei->first == nodeId) {
            continue;
        }
        if(nodes[nei->first].cluster == origClusterId) {
            modLeaving -= nei->second;         
        }
    }
    // baseline degrees, excluding the node itself
    for(auto ni2=origCluster->nodes.begin(); ni2 != origCluster->nodes.end(); ni2++) {
        if((*ni2) == nodeId) {
            continue;
        }
        // subnetworks use global degrees
        if(isSubnetwork) {
            modLeaving += node.degreeGlobal * nodes[*ni2].degreeGlobal / twiceTotalEdgesGlobal;
        } else {
            modLeaving += node.degree * nodes[*ni2].degree / twiceTotalEdges;
        }
    }

    for(auto c=clusters.begin(); c != clusters.end(); c++) {
        if(std::find(neighboringClusters.begin(), neighboringClusters.end(), c->clusterId) == neighboringClusters.end())
            continue;
        // skip the original cluster, it's done after the for loop
        if(c->clusterId == origClusterId)
            continue;

        // Leaving your cluster has a fixed difference in modularity
        double mod = modLeaving;

        // The cluster gains a few internal links (with nodeId)
        // The original cluster might lose a few links
        // vice versa with the squared sums of degrees
        // so there are 4 terms in this evaluation

        // Check additional edges into this cluster, excluding self edge
        // we don't need to do anything about the self edge, since it's a different cluster
        for(auto nei=node.neighbors.begin(); nei != node.neighbors.end(); nei++) {
            if(nodes[nei->first].cluster == c->clusterId){
                mod += nei->second;         
            }
        }
        // Subtract k_i sum_j k_j / (2m)^2 from the new cluster
        // no need to worry about self weight, it was not added to start with
        for(auto ni2=c->nodes.begin(); ni2 != c->nodes.end(); ni2++) {
            // subnetworks use global weights
            if(isSubnetwork) {
                mod -= node.degreeGlobal * nodes[*ni2].degreeGlobal / twiceTotalEdgesGlobal;
            } else {
                mod -= node.degree * nodes[*ni2].degree / twiceTotalEdges;
            }
        }

        if(mod > modMax) {
            modMax = mod;
            clusterIdMax = c->clusterId;
        }
    }
    
    // finally, try to make a new singleton cluster with this node
    // no edges except self-edge, but that was not added to start with
    // self weight comes back but was not added to start with 
    // conclusion: mod = modLeaving
    if(modLeaving > modMax) {
        // figure the first free clusterId
        // this is a form of recycling
        std::set<uint64_t> clusterIds;
        for(auto c=clusters.begin(); c != clusters.end(); c++) {
            clusterIds.insert(c->clusterId);
        }
        clusterIdMax = 0;
        while(clusterIds.find(clusterIdMax) != clusterIds.end()) {
            clusterIdMax++;
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

bool Network::runLocalMovingAlgorithm() {
    bool update = false;
    if(nodes.size() == 1)
        return update;

#if SLMPY_VERBOSE
    std::cout << "is subnetwork: " << isSubnetwork << std::endl << std::flush;
#endif

    std::vector<uint64_t> nodesShuffled = nodesInRadomOrder();

    uint64_t numberStableNodes = 0;
    uint64_t  i = 0;
    uint64_t nodeId;
    uint64_t bestClusterId;
    bool isStable;
    int64_t iteration = 0;
    int64_t maxIteration = -1;
    do {
#if SLMPY_VERBOSE
        std::cout << "runLocalMovingAlgorithm, iteration " << (iteration + 1) << std::endl << std::flush;
#endif

        isStable = false;
        nodeId = nodesShuffled[i];

#if SLMPY_VERBOSE
        std::cout << "findBestCluster" << std::endl << std::flush;
#endif
        // fixed nodes never change
        if(fixedNodes.find(nodeId) != fixedNodes.end()) {
            bestClusterId = nodes[nodeId].cluster;
        } else {
            // Find best cluster for the random node, including its own one
            bestClusterId = findBestCluster(nodeId);
        }
#if SLMPY_VERBOSE
        std::cout << "bestClusterId: " << bestClusterId << std::endl << std::flush;
#endif

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
        if(iteration == maxIteration)
            break;
    
    } while(numberStableNodes < nodes.size());

    return update;
}


bool Network::runLouvainAlgorithm() {
    bool update = false;
    bool update2;
    if(nodes.size() == 1)
        return update;

    update |= runLocalMovingAlgorithm();

    if(clusters.size() == nodes.size())
        return update;

    Network redNet = calculateReducedNetwork();
    redNet.createSingletons();
    update2 = redNet.runLouvainAlgorithm();

#if SLMPY_VERBOSE
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
#endif

    if(update2) {
        update = true;
        mergeClusters(redNet.clusters);
    }
    return update;
}


bool Network::runSmartLocalMovingAlgorithm() {
    bool update = false;
    if(nodes.size() == 1)
        return update;

    update |= runLocalMovingAlgorithm();

    if(clusters.size() == nodes.size())
        return update;
    
    // each community -> subnetwork, reset labels and runLocalMoving inside that subnetwork
    // then set the cluster ids as of the double splitting
    uint64_t nClusters = 0;
    uint64_t isn = 0;
    std::map<uint64_t, uint64_t> clusterToSubnetwork;
    for(auto c=clusters.begin(); c != clusters.end(); c++, isn++) {
#if SLMPY_VERBOSE
        std::cout << "subnetwork " << isn+1 <<std::endl << std::flush;
#endif

        // create subnetwork
        Network subNet = createSubnetwork(c);

        // cluster within subnetwork
        subNet.runLocalMovingAlgorithm();

        // assign community numbers to all nodes across the parent network
        std::map<uint64_t, uint64_t> clusterMap;
        uint64_t i = 0;
        for(auto c1=subNet.clusters.begin(); c1 != subNet.clusters.end(); c1++, i++) {
            clusterMap[c1->clusterId] = i;
        } 
        for(auto n=subNet.nodes.begin(); n!=subNet.nodes.end(); n++) {
            // n->cluster has the clusterId within the subnetwork, go to the
            // global list of nodes and update it with a new clusterId that
            // runs over all subnetworks
            nodes[n->first].cluster = nClusters + clusterMap[n->second.cluster];
        }
        // make a list of which of the final clusters/reduced nodes belongs
        // to which subnetwork for later initialization of the reduced network
        for(auto c1=subNet.clusters.begin(); c1 != subNet.clusters.end(); c1++) {
            clusterToSubnetwork[nClusters++] = isn; 
        }

#if SLMPY_VERBOSE
        std::cout << "clusterToSubnetwork: ";
        for(auto c1=clusterToSubnetwork.begin(); c1 != clusterToSubnetwork.end(); c1++) {
            std::cout << "(" << c1->first << ", " << c1->second << ") ";
        }
        std::cout << std::endl << std::flush;
#endif

    }

    // recalculate clusters
    calcClustersFromNodes();

    Network redNet = calculateReducedNetwork();

    // the initial state is not each reduced node for itself, but rather
    // each subnetwork for itself. This is probably for convergence/speed reasons
    redNet.createFromSubnetworks(clusterToSubnetwork);

    update |= redNet.runSmartLocalMovingAlgorithm();

    mergeClusters(redNet.clusters);

    return update;
}



// the next two functions go up and down the subnetwork business
void Network::createFromSubnetworks(std::map<uint64_t, uint64_t> clusterToSubnetwork) {
    // instead of setting each (reduced) node to its own community
    // we give them the same cluster if they belong to the same subnetwork
    for(auto n=nodes.begin(); n != nodes.end(); n++) {
        // reduced network initialization puts every reduced node to the clusterId they came from
        n->second.cluster = clusterToSubnetwork[n->second.cluster];
    }

    calcClustersFromNodes();
    calcDegreesAndTwiceTotalEdges();
}

Network Network::createSubnetwork(std::vector<Cluster>::iterator c) {

    Network subNet;
    subNet.isSubnetwork = true;

    // put in the nodes (pointers would be better but ok)
    // FIXME: it is unclear from the paper whether global degrees
    // percolate up to the very top or stop at the parent. This
    // is relevant for subnetworks of reduced networks
    uint64_t cId = 0;
    for(auto n=c->nodes.begin(); n != c->nodes.end(); n++, cId++) {
        Node node(cId);
        node.degreeGlobal = nodes[*n].degree;
        // only keep neighbors within the subnetworks
        for(auto nei=nodes[*n].neighbors.begin(); nei != nodes[*n].neighbors.end(); nei++) {
            if(nodes[nei->first].cluster == nodes[*n].cluster) {
                node.neighbors[nei->first] = nei->second;
            }
        }
        subNet.nodes[*n] = node;
        // start with singletons
        subNet.clusters.push_back(Cluster(cId, {*n}));
    }
    subNet.twiceTotalEdgesGlobal = twiceTotalEdges;

    return subNet;
}
