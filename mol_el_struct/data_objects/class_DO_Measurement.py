
from data_objects.class_Data_Object import Data_Object
from utils.class_Disk_Jockey import Disk_Jockey


class DO_Measurement(Data_Object):

    object_type = "measurements"
    data_nodes = {
        "context" : { # Provides the context for the measurement: specifies molecule and sampling method, as well as data group label
                "system" : "json", # molecule label, data group label
                "method" : "json" # method label and kwargs
            }
        "basis" : { # can be used to reconstruct CS_sample at the time of final iteration
                "z_params" : "csv", # param vector for each basis state, as obtained by sampling
                "overlaps" : "json" # S and H matrices with state index given by its position in z_params
            },
        "results" : { # eigenvals and eigenstates of H given S
                "E_min" : "csv", # number of states, min energy
                "u_min" : "csv" # number of states, coefs of first N basis states
            }
        }

    def __init__(self, ID, j = None):
        # ID is the identification label
        # j is a Journal instance passed on to DJ
        super().__init__(ID, j)
