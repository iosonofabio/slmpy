import numpy as np



class ModularityOptimzer:

    def __init__(self):
        pass


    @classmethod
    def load_from_edge_list(cls, edges):
        '''Load a ModularityOptimzer from a list of edges

        Args:
            edges (list of pairs or Nx2 np.ndarray): the list of edges
        '''

        self = cls()
        self.edges = np.asarray(edges, dtype=np.uint64)
        return self
