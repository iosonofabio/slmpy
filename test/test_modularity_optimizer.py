# vim: fdm=indent
'''
author:     Fabio Zanini
date:       01/04/19
content:    Test ModularityOptimizer
'''
import sys
import numpy as np
import pytest


def test_load_edges():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer(
            [
             [0, 1],
             [3, 1],
            ],
            )
    assert(np.allclose([[0, 1], [3, 1]], mo.edges))


def test_load_karate():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_tsv_file(
            'data/karate_club.tsv',
            )
    assert(np.allclose([[0, 1], [0, 2]], mo.edges[:2]))


def test_call_interface_zero_iterations():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer(
            [
             [0, 1],
             [3, 1],
            ],
            )
    mo.n_iterations = 0
    mo.communities = np.arange(3, dtype=np.uint64)
    a = mo(algorithm='local_moving')
    assert((a == [0, 1, 2]).all())


def test_call_interface_local_heuristic():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer(
            [
             [0, 1],
             [0, 2],
             [1, 2],
            ],
            )
    mo.n_iterations = 1
    a = mo(algorithm='local_moving')
    assert((a == a[0]).all())


def test_double_diamond_louvain():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer(
            [
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
            ],
            )
    mo.n_iterations = 1
    a = mo(algorithm='louvain')

    answer = [0, 0, 0, 0, 1, 1, 1]

    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


def test_double_diamond_louvain_fixed():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer(
            [
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
            ],
            )
    mo.n_iterations = 1
    mo.fixed_nodes = [0, 4]
    a = mo(algorithm='louvain')

    answer = [0, 0, 0, 0, 1, 1, 1]

    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


def test_double_diamond_slm_fixed():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer(
            [
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
            ],
            )
    mo.n_iterations = 1
    mo.fixed_nodes = [0, 4]
    a = mo(algorithm='smart_local_moving')

    answer = [0, 0, 0, 0, 1, 1, 1]

    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


def test_double_diamondplus():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer(
            [
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
            ],
            )
    mo.n_iterations = 1
    a = mo(algorithm='louvain')

    answer = [0, 0, 0, 0, 1, 1, 1, 2, 2]

    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


@pytest.mark.xfail(True, reason='Brittle test, depends on random number generator')
def test_call_interface_karate_louvain():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_tsv_file(
            'data/karate_club.tsv',
            )
    # NOTE: louvain gets 2 nodes wrong
    answer = np.loadtxt(
            'data/karate_club_communities_louvain.tsv',
            dtype=np.uint64,
            )
    mo.n_iterations = 1

    a = mo(algorithm='louvain')
    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


# turns out iterating louvain works
def test_call_interface_karate_louvain_iterative():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_tsv_file(
            'data/karate_club.tsv',
            )
    answer = np.loadtxt(
            'data/karate_club_communities.tsv',
            dtype=np.uint64,
            )
    mo.n_iterations = 3

    a = mo(algorithm='louvain')
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


def test_call_interface_karate_louvain_fixed():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_tsv_file(
            'data/karate_club.tsv',
            )
    # NOTE: louvain gets 2 nodes wrong, what if we fix some?
    mo.communities[25] = 23
    mo.communities[23] = 23
    for node, com in zip(mo.nodes, mo.communities):
        print(int(node+1), int(com+1))
    mo.fixed_nodes = np.array([23, 25], np.uint64)
    answer = np.loadtxt(
            'data/karate_club_communities.tsv',
            dtype=np.uint64,
            )
    mo.n_iterations = 2

    a = mo(algorithm='louvain')
    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())
    # Answer: it fixes it


@pytest.mark.xfail(True, reason='Brittle test, depends on random number generator')
def test_call_interface_karate_slm():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_tsv_file(
            'data/karate_club.tsv',
            )
    answer = np.loadtxt(
            'data/karate_club_communities.tsv',
            dtype=np.uint64,
            )
    mo.n_iterations = 1

    a = mo(algorithm='smart_local_moving')
    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


@pytest.mark.xfail(True, reason='Brittle test, depends on random number generator')
def test_call_interface_karate_slm_iterative():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_tsv_file(
            'data/karate_club.tsv',
            )
    answer = np.loadtxt(
            'data/karate_club_communities.tsv',
            dtype=np.uint64,
            )
    mo.n_iterations = 2

    a = mo(algorithm='smart_local_moving')
    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


@pytest.mark.xfail(True, reason='Brittle test, depends on random number generator')
def test_call_interface_karate_slm_fixed():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_tsv_file(
            'data/karate_club.tsv',
            )
    answer = np.loadtxt(
            'data/karate_club_communities.tsv',
            dtype=np.uint64,
            )
    # try fixing nodes in SLM
    # FIXME
    mo.fixed_nodes = np.array([0, 6, 33], np.uint64)
    mo.n_iterations = 5
    a = mo(algorithm='smart_local_moving')
    print([int(str(x)[-1]) for x in np.arange(34) + 1])
    print(list(answer))
    print(list(a))
    assert((a == answer).all())


#def test_call_interface_slm():
#    from slmpy import ModularityOptimzer
#
#    mo = ModularityOptimzer(
#            [
#             [0, 1],
#             [0, 2],
#             [1, 2],
#            ],
#            )
#    mo.n_iterations = 1
#    a = mo(algorithm='smart_local_moving')
#    assert((a == a[0]).all())
#
#
