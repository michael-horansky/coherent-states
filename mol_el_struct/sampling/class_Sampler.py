# -----------------------------------------------------------------------------
# ------------------------------- class Sampler -------------------------------
# -----------------------------------------------------------------------------
# This class creates an object which can sample random numbers from certain PD
# distributions using named algorithms.
# It can be passed to iterative methods which create samples of physical
# objects, such as CS samples.

import numpy as np

import functions

from project_management_utils.color import color
from project_management_utils.Correspondent import Correspondent

class Sampler:

    methods = {
            "MNDp" : {
                    "title" : "Multivariate normal distribution (phase randomisation)",
                    "desc" : "A multi-dimensional Gaussian distribution constructed from given means, variances, and covariances. Diagonalises the covariance matrix to generate independent variables and then reconstructs the original variables. Clips non-positive eigenvalues to ensure safe sampling.",
                    "properties" : {
                            "m" : "List of means of the N random variables. N is inferred from len(m).",
                            "S" : "Covariance matrix of the N random variables. Shape must be (len(m), len(m)).",
                            "pr" : "Phase randomisation for parameter tensor. \"ai\": as is. \"rs\": random sign. \"rp\": random complex phase."
                        }
                },
            "MNDps" : {
                    "title" : "Multivariate normal distribution with spin-subspace symmetrisation",
                    "desc" : "A multi-dimensional Gaussian distribution constructed from given means, variances, and covariances. Diagonalises the covariance matrix to generate independent variables and then reconstructs the original variables. Clips non-positive eigenvalues to ensure safe sampling. Enforces a certain amount of symmetry between the two spin-subspaces.",
                    "properties" : {
                            "m" : "List of means of the N random variables. N is inferred from len(m). Unless sym is 'none', N only covers one spin-subspace, and thus the output is doubled in dimension.",
                            "S" : "Covariance matrix of the N random variables. Shape must be (len(m), len(m)).",
                            "pr" : "Phase randomisation for parameter tensor. \"ai\": as is. \"rs\": random sign. \"rp\": random complex phase.",
                            "sym" : "Symmetrisation of parameter tensor across spin-subspaces. \"full\": tensors equal. \"phase\": elementwise magnitudes equal. \"none\": no symmetrisation."
                        }
                }
        }

    def __init__(self, method, journal_instance = None, **kwargs):
        # method [string] is a keyword which specifies what sampling method (family of PDFs) is used.
        # It must be found in Sampler.methods.keys().
        # kwargs are properties of the PDF, stored in a dictionary self.properties. All keywords listed in the documentation of the method must be present.

        self.log = Correspondent(journal_instance, {"ID" : f"S_{method}", "label" : f"Sampler ({method})", "color" : color.BRIGHT_MAGENTA})

        if method not in Sampler.methods.keys():
            self.log.error(f"Method {method} not a valid method. Sampler cannot be initialised.")
            self.print_methods()
            quit()

        self.method = method
        self.properties = kwargs

        for req_prop in Sampler.methods[self.method]["properties"].keys():
            if req_prop not in self.properties.keys():
                self.log.error(f"Missing property: '{req_prop}' ({Sampler.methods[self.method]['properties'][req_prop]}).")
                quit()

        self.log.write(f"Sampler successfully initialised. Chosen method: {Sampler.methods[self.method]['title']}.")

        self.sampling_methods = {
            "MNDp" : self.get_sample_MNDp,
            "MNDps" : self.get_sample_MNDps
        }

    def print_methods(self):
        # Needs to be an instance method since it uses the provided journal
        self.log.write("Available sampling methods:")
        self.log.print_itemize(Sampler.methods)


    def get_sample(self, N = 1):
        # N is the size of the sample.
        # The output is an (N, ...) array, where the remaining elements of the shape are determined by the sampling method

        if N > 1:
            res = []
            for n in range(N):
                res.append(self.get_sample())
            return(res)

        return(self.sampling_methods[self.method]())

    # --------------------------- Sampling methods ----------------------------

    def get_sample_MNDp(self):

        # We obtain the random, normally-distributed Z
        eigvals, eigvecs = np.linalg.eigh(self.properties["S"])
        eigvals = np.clip(eigvals, 0, None)
        L = eigvecs * np.sqrt(eigvals)[None, :]
        randvar = np.random.randn(self.properties["S"].shape[0])
        Z_real = randvar @ L.T + self.properties["m"]

        # We then apply phase randomisation
        Z = Z_real * functions.randsign_mask((self.properties["S"].shape[0],), self.properties["pr"])


    def get_sample_MNDps(self):


        # We obtain the random, normally-distributed Z
        eigvals, eigvecs = np.linalg.eigh(self.properties["S"])
        eigvals = np.clip(eigvals, 0, None)
        L = eigvecs * np.sqrt(eigvals)[None, :]
        randvar = np.random.randn(self.properties["S"].shape[0])
        Z_real = randvar @ L.T + self.properties["m"]

        # We then apply phase randomisation
        Z = Z_real * functions.randsign_mask((self.properties["S"].shape[0],), self.properties["pr"])

        if self.properties["sym"] == "full":
            # We simply copy the sample into the other spin-subspace

            return(np.array([Z, Z.copy()]))

        elif self.properties["sym"] == "phase":
            # We copy the magnitudes of Z into the other spin-subspace and then randomise their phases

            return(np.array([Z, Z.copy() * functions.randsign_mask((self.properties["S"].shape[0],), "rp")]))


        elif self.properties["sym"] == "none":
            # No symmetrisation. Z is the output we wanted.
            return(Z)

            #return(np.array([Z[:int(len(Z) / 2)], Z[int(len(Z) / 2):]]))


