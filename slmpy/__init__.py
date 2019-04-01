import numpy as np



class ModularityOptimzer:

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
        if self.edges.shape[1] != 2:
            raise ValueError('Edges must be pairs')
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
        if self.edges.shape[1] != 2:
            raise ValueError('Edges must be pairs')
        return self

