[![Build Status](https://travis-ci.org/iosonofabio/slmpy.svg?branch=master)](https://travis-ci.org/iosonofabio/slmpy)

# slmpy
Smart Local Moving community detection algorithm in Python and C++11/14, with fixed points.

## Usage
More docs will follow, for now here's an example:

```python
    from slmpy import ModularityOptimzer

    edges = [
             [0, 1],
             [0, 2],
             [0, 3],
             [1, 2],
             [1, 3],
             [2, 3],
             [3, 4],
             [4, 5],
             [4, 6],
             [5, 6],
             [7, 8],
            ]

    mo = ModularityOptimzer(edges)
    mo.n_iterations = 1
    mo.fixed_nodes = [0, 4]  # This fixes nodes 0 and 4 to be in different communities
    a = mo(algorithm='smart_local_moving')

    # Check answer
    assert((a == [0, 0, 0, 0, 1, 1, 1]).all())
```

## References
The original Smart Local Moving algorithm is outlined in the *excellent* article:

  [Ludo Waltman](http://www.ludowaltman.nl/) and Nees Jan van Eck, "A smart local moving algorithm for large-scale modularity-based community detection", Eur. Phys. J. B (2013) 86: 471. DOI: [10.1140/epjb/e2013-40829-0](http://dx.doi.org/10.1140/epjb/e2013-40829-0)

The code is based on the wonderful Java version by deepminder: https://github.com/deepminder/SLM4J .
