# vim: fdm=indent
'''
author:     Fabio Zanini
date:       01/04/19
content:    Test ModularityOptimizer
'''
import sys
import numpy as np
import pytest



def test_init():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer()


def test_load_edges():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_list(
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


def test_call_interface():
    from slmpy import ModularityOptimzer

    mo = ModularityOptimzer.load_from_edge_list(
            [
             [0, 1],
             [3, 1],
            ],
            )
    mo()
