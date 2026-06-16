# Each object is a simple header which specifies the arguments for a given
# Access Point (AP) script. It handles typing and input errors.

import sys


class AccessPoint():

    def __init__(self, name, desc, params):
        # name is a short string; the filename of the script
        # desc is a long description of what the script does
        # params is a list in the form [i] = [label, type, default value, desc]
        #     -label is a short string; the name of the parameter
        #     -type is a class, and the input gets re-typed using the provided
        #      class
        #     -if the parameter is missing and default value is not None, the
        #      default value is used instead. The general structure must adhere
        #      to [mandatory arguments ... optional arguments ...]
        #     -desc is a string, and describes the purpose of the parameter
        self.name = name
        self.desc = desc
        self.params = params

    def process_cmd_args(self):
        # Returns a list of properly-typed arguments if cmd_args is sane
        # Raises an error and returns None if cmd_args is ill-formed

        cmd_args = sys.argv[1:]
        N_a = len(cmd_args)
        N_p = len(self.params)

        res = []

        print("Processing command parameters...")

        if N_a > N_p:
            raise Exception(f"Too many parameters provided ({N_p} expected; {N_a} provided)")

        for i in range(N_a):
            try:
                res.append(self.params[i][1](cmd_args[i]))
            except:
                raise Exception(f"Conversion of parameter \"{self.params[i][0]}\" failed (type \"{str(self.params[i][1])}\" expected; provided value {cmd_args[i]} is of type {typeof(cmd_args[i])})")
            print(f"  \"{self.params[i][0]}\" set to {res[i]}")

        for i in range(N_a, N_p):
            if self.params[i][2] is None:
                raise Exception(f"Parameter \"{self.params[i][0]}\" mandatory but not provided (parameter index {i})")
            else:
                res.append(self.params[i][2])
                print(f"  \"{self.params[i][0]}\" defaulted to {res[i]}")
        print("  Success!")

    def man(self):
        # Prints the manual for the access point, i.e. the required parameters
        print("This is a computation access point.")
        print(f"Script name: {self.name}")
        print(self.desc)
        print("-" * 80)
        print("Parameters:")
        for i in range(len(self.params)):
            if self.params[i][2] is None:
                print(f"  \"{self.params[i][0]}\" [{str(self.params[i][1])}]: {self.params[i][3]}")
            else:
                print(f"  \"{self.params[i][0]}\" [{str(self.params[i][1])}] (default {str(self.params[i][2])}): {self.params[i][3]}")




########################### ACCESS POINT CATALOGUE ############################

sep_coef_AP = AccessPoint(
        "Separation coefficient variation",
        "For a given diatomic molecule, calculates its electronic ground state energy for different separation coefficients. The basis sample parameter tensor may be frozen to avoid re-sampling for each separation coefficient value.",
        [
            ["mol", str, None, "Molecule label. Must exist as a key in molecules_abstract.mol_catalogue"],
            ["N", int, None, "Size of basis sample (sans ref state) in the calculation. Time complexity is quadratic in N."],
            ["N_sub", int, None, "Size of subsample for each basis state candidate. Time complexity is linear in N_sub."],
            ["sr", str, "rp", "Sign randomisation for parameter tensor. \"ai\": as is. \"rs\": random sign. \"rp\": random complex phase."],
            ["freeze_basis", int, 1, "Whether the parameter tensor is frozen based on the sampling for the default separation coefficient. 1: frozen. 0: not frozen (much higher time complexity)."]
        ]
    )


