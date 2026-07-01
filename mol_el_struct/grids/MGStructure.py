


# Parent class. Not specific. Can be used to create arbitrary, unstructured grids

class MGStructure():

    def __init__(self, mol_bp, base_g, geometries, connections = None):
        # mol_bp is an instance of MBlueprint
        # base_g is an instance of MGeometry
        # geometries is a list of instances of MGeometry
        # connections is a list of the form [canonical index of g] = [list of canonical indices of connected gs]
        self.mol_bp = mol_bp
        self.base_can_i = 0

        self.geometries = geometries
        self.nodes = []
        for i in range(len(self.geometries)):
            self.nodes.append(MPSNode(self.mol_bp, self.geometries[i]))

        if connections is None:
            self.create_automatic_connections()
        else:
            self.connections = connections

        # Metadata
        self.geometry_meta = {
            "class" : "MGStructure",
            "desc" : "Manual initialisation.",
            "mol_bp" : self.mol_bp.mol_meta
            }

        self.base_node_meta = {
            "N_surf" : None,
            "base_can_i" : self.base_can_i
            }

        self.nodes_meta = {
            "HF_method" : None,
            "FCI_known" : False,
            "surface_labels" : []
            }



    def get(self, i):
        # This method is used to create a natural indexing, here translated to canonical indexing. Subclasses will redefine this.
        if i >= 0 and i < len(self.nodes):
            return(i)
        return(None)

    def find(self, i):
        # This method translates canonical index to natural index
        return(i)

    def neighbours(self, i):
        # returns canonical indices of all neighbours of node at i
        return(self.connections[i])

        # NOTE: when making subclasses, we should skip using a connections list
        # if possible, as it requires a lot of memory. Regular structure often
        # allows us to directly overload this method.

    def create_automatic_connections(self):
        # Starting from base_g, we connect each geometry to all of its closest yet untraversed geometries
        pass
