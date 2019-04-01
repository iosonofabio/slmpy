#include <map>
#include <cmath>
#include <slmpy.h>


uint64_t Node::degree() {
    return (uint64_t)neighbors.size();
}
