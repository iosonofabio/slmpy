import numpy as np
from ._slmpy import smart_local_moving as _smart_local_moving



class ModularityOptimzer:

    n_iterations = 1

    def __init__(self):
        pass


    @classmethod
    def load_from_edge_list(cls, edges):
        '''Load a ModularityOptimzer from a list of edges

        Args:
            edges (list of pairs or Nx2 np.ndarray): the list of edges

        Returns:
            ModularityOptimzer with those edges.
        '''
        self = cls()
        self.edges = np.asarray(edges, dtype=np.uint64)
        self._finish_initialization()
        return self

    @classmethod
    def load_from_edge_tsv_file(cls, filename):
        '''Load a ModularityOptimzer from a TSV filename with the edges

        Args:
            filename (str): the path to the file containing the edges, which
            must be in TSV (tab separated values) format, in which each row is
            an edge between the first and second colunm node.

        Returns:
            ModularityOptimzer with edges as in the file.
        '''
        self = cls()
        self.edges = np.loadtxt(filename, delimiter='\t', dtype=np.uint64)
        self._finish_initialization()
        return self

    def _finish_initialization(self):
        if self.edges.shape[1] != 2:
            raise ValueError('Edges must be pairs')

        self.nodes = np.unique(self.edges)

        # The default is to start with each node in its own community
        # NOTE: the communities are numbered 0 to N-1 by default
        self.communities = np.arange(len(self.nodes), dtype=np.uint64)

    def __call__(self, random_seed=0):
        '''Run the smart local moving algorithm'''

        communities_out = self.communities.copy()
        _smart_local_moving(
                communities_out,
                self.edges,
                self.nodes,
                self.communities,
                random_seed,
                self.n_iterations,
                )
        return communities_out
