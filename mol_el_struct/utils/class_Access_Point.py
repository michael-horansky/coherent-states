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

        self.param_names = []
        for i in range(len(self.params)):
            self.param_names.append(self.params[i][0])

    def enforce_type(self, i, val):
        try:
            return(self.params[i][1](val))
        except:
            raise Exception(f"Conversion of parameter \"{self.params[i][0]}\" failed (type \"{str(self.params[i][1])}\" expected; provided value {val} is of type {type(val)})")

    def process_cmd_args(self):
        # Returns a list of properly-typed arguments if cmd_args is sane
        # Raises an error and returns None if cmd_args is ill-formed

        cmd_args = sys.argv[1:]

        res = []

        print("Processing command parameters...")

        # We split the positional and keyword arguments

        args_pos = []
        args_kw = {}

        initialised_params = []

        kw_arg_encountered = False
        for i in range(len(cmd_args)):
            if "=" in cmd_args[i]:
                key, value = cmd_args[i].split("=", 1)
                if key not in self.param_names:
                    raise Exception(f"Unknown keyword argument \"{key}\" (provided value {value}).")
                param_index = self.param_names.index(key)
                if param_index in initialised_params:
                    raise Exception(f"Keyword argument \"{key}\" has encountered duplicate value assignment.")
                initialised_params.append(param_index)
                args_kw[key] = value
                kw_arg_encountered = True
            elif kw_arg_encountered:
                raise Exception(f"Positional argument (i. {i + 1}, val. {cmd_args[i]}) encountered after a keyword argument. Please use the format [positional arguments] [keyword arguments].")
            else:
                args_pos.append(cmd_args[i])
                initialised_params.append(i)


        for i in range(len(args_pos)):
            res.append(self.enforce_type(i, args_pos[i]))
            print(f"  \"{self.params[i][0]}\" set to {res[i]}")

        for i in range(len(args_pos), len(self.params)):

            if self.params[i][0] in args_kw.keys():
                res.append(self.enforce_type(i, args_kw[self.params[i][0]]))
                print(f"  \"{self.params[i][0]}\" set to {res[i]}")

            elif self.params[i][2] is None:
                raise Exception(f"Parameter \"{self.params[i][0]}\" mandatory but not provided (parameter index {i}).")
            else:
                res.append(self.params[i][2])
                print(f"  \"{self.params[i][0]}\" defaulted to {res[i]}")
        print("  Success!")
        return(res)

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


