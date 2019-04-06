import numpy as np
from ._slmpy import smart_local_moving, local_moving, louvain



class ModularityOptimzer:

    def __init__(
            self,
            edges,
            nodes=None,
            communities=None,
            fixed_nodes=tuple(),
            n_iterations=1,
            ):
        '''ModularityOptimizer

        Args:
            edges (list of pairs): edges in the graph
            nodes (list): nodes in the graph. If it is None, they are the
            unique list of nodes found in the edges.
            communities (list): initial community assignment. If it is None,
            each node is a singleton.
            fixed_nodes (list): list of nodes to fix throughout the clustering
            algorithm.
        '''
        if nodes is None:
            nodes = np.unique(edges)
        if communities is None:
            communities = np.arange(len(nodes), dtype=np.uint64)

        self.edges = edges
        self.nodes = nodes
        self.communities = communities
        self.fixed_nodes = fixed_nodes
        self.n_iterations = n_iterations

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, values):
        self._nodes = np.asarray(values, dtype=np.uint64)

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, values):
        self._edges = np.asarray(values, dtype=np.uint64)

    @property
    def communities(self):
        return self._communities

    @communities.setter
    def communities(self, values):
        self._communities = np.asarray(values, dtype=np.uint64)

    @property
    def fixed_nodes(self):
        return self._fixed_nodes

    @fixed_nodes.setter
    def fixed_nodes(self, values):
        self._fixed_nodes = np.asarray(values, dtype=np.uint64)

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
        edges = np.loadtxt(filename, delimiter='\t', dtype=np.uint64)
        return cls(edges=edges)

    def __call__(self, algorithm='smart_local_moving', random_seed=0):
        '''Run the optimizer

        Args:
            algorithm (str): one of "local_moving", "lovain", and
            "smart_local_moving".

        Returns:
            one dimensional numpy.ndarray: the communities the nodes belong to.

        NOTE: if the nodes were not explicitely set, they can be recovered via
        the nodes property.
        '''

        communities_out = self.communities.copy()

        if algorithm == 'smart_local_moving':
            smart_local_moving(
                    communities_out,
                    self.edges,
                    self.nodes,
                    self.communities,
                    self.fixed_nodes,
                    random_seed,
                    self.n_iterations,
                    )
        elif algorithm == 'louvain':
            louvain(
                    communities_out,
                    self.edges,
                    self.nodes,
                    self.communities,
                    self.fixed_nodes,
                    random_seed,
                    self.n_iterations,
                    )
        elif algorithm == 'local_moving':
            local_moving(
                    communities_out,
                    self.edges,
                    self.nodes,
                    self.communities,
                    self.fixed_nodes,
                    random_seed,
                    self.n_iterations,
                    )

        return communities_out
