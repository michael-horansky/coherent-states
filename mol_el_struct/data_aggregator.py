# -----------------------------------------------------------------------------
# ------------------------------ Data Aggregator ------------------------------
# -----------------------------------------------------------------------------
# This object loads data produced by various solvers; the data can be collected
# dynamically from a user-specified parameter space. Methods are provided for
# aggregating along parameter space dimensions and plotting parameter surfaces.

class DataAggregator():

    def __init__(self, ID):

        self.ID = ID

        self.parameters = {}

        self.filename_convention = None


    def add_parameter_range(self, param, values):
        self.parameters[param] = values

    def set_filename_convention(self, filename_convention):
        self.filename_convention = filename_convention


    def get_filename(self, param_values):
        # param_values is a dict {param : value}

        cur_filename = self.filename_convention

        for param in self.parameters:
            cur_filename.replace(f"%{param}", str(param_values[param]))

        return(cur_filename)


