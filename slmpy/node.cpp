#include <map>
#include <cmath>
#include <slmpy.h>


// edges are weighted, so the degree of a node is the
// sum of the edge weights
double Node::degree() {
    double deg = 0;
    for(auto n=neighbors.begin(); n != neighbors.end(); n++)
        deg += n->second;
    return deg;
}
