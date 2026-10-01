# -----------------------------------------------------------------------------
# ------------------------------- class Operator ------------------------------
# -----------------------------------------------------------------------------
# An instance of Operator represents an arbitrary operator expressed using fer.
# annihilation and creation operators. It is represented by a lin. combination
# of elementary normal-ordered fermionic operators, each one equivalent to two
# integer sets: the set of indices of the creation operators and the set of
# indices of the annihilation operators. The elementary operator is understood
# to be monotone-ordered like so:
#
# ({1, 2, 3}, {1, 2, 3}) <=> f_1\hc f_2\hc f_3\hc f_3 f_2 f_1
#
# The purpose of this class is to prepare frequently-used operators before the
# main computation commences, to save time by performing a single complex
# process. Operators can be constructed by addition or multiplication, the
# latter using Wick's theorem.

import numbers
import numpy as np

# Auxiliary functions

# Converts parity to sign
def sign(i):
    if i % 2 == 0:
        return(1.0)
    else:
        return(-1.0)

def is_number(x):
    return(isinstance(x, numbers.Number) and not isinstance(x, bool))

def has_imag_comp(x):
    assert is_number(x)
    if isinstance(x, (complex, np.complexfloating)):
        if x.imag != 0.0:
            return(True)
    return(False)

def has_real_comp(x):
    assert is_number(x)
    if isinstance(x, (complex, np.complexfloating)):
        if x.real != 0.0:
            return(True)
        return(False)
    if x == 0.0:
        return(False)
    return(True)

# Iterator which iterates through 2^N bitmasks on N variables, represented by a list of N true/false bools
class binit:
    def __init__(self, N, invert = False, reverse = False):
        self.N = N
        self.invert = invert
        self.reverse = reverse
        self.state = [not self.invert] * N
        self.head = False # did we return the Head already?

    def __iter__(self):
        return self

    def __next__(self):
        if not self.reverse:
            cur_range = range(self.N)
        else:
            cur_range = range(self.N - 1, -1, -1)
        for i in cur_range:
            if self.state[i] == self.invert:
                self.state[i] = not self.invert
                return self.state
            else:
                self.state[i] = self.invert
        if self.head:
            raise StopIteration
        else:
            self.head = True
            return self.state





class Operator():

    @classmethod
    def zero(cls):
        # returns the zero (empty) operator
        return(cls({}))

    @classmethod
    def copy(cls, op):
        res = cls.zero()
        for key in op.phi:
            res.phi[key] = op.phi[key]
        return(res)

    @classmethod
    def normal_order(cls, expression, coef = 1):
        # Here, expression is a list of operators in the form ('c'/'a', i)
        # where the first tuple element specifies a creation or annihilation o.
        # and the second tuple element specifies the mode index.

        # Modes are indexed from zero

        # This class method returns a new Operator which is equal to the
        # provided expression

        # We do not actually use Wick's theorem, since its naive implementation
        # is actually very computationally complex. Practically, we would have
        # to consider the possible contractions for each mode index i, and then
        # take the massive direct product of all the possible sets of
        # contractions across all i = 1 ... N. Lots of trivially zero terms
        # would occupy computation time (especially due to a repeated operator
        # of the form f_i ... f_i or f_i\hc ... f_i\hc).

        # Instead, we apply the theorem to each fixed-mode-index substring,
        # classifying the "skippable" spaces by whether their mode indices are
        # smaller or larger than that of the current substring, so that we can
        #   a) keep proper track of the Jordan string, and
        #   b) quickly arrive at the monotone-ordered expression without
        #      explicitly calculating permutation parities
        # Note that the substring may be odd-length. The only SU(M) restriction
        # is on the total number of 'c' and 'a' operators.

        # Max mode index
        M = 0
        for i in range(len(expression)):
            if expression[i][1] + 1 > M:
                M = expression[i][1] + 1 # +1 due to modes being indexed from zero

        # Number of operators per mode.
        N = [0] * M

        # Signature of each substring, plus helping methods
        signature = [None] * M # If left at None, it means the substring was not present at all
        first_op_type = [None] * M
        cur_op_type = [None] * M

        # Conditional list indices for each mode index
        C = []
        A = []
        for i in range(M):
            C.append([])
            A.append([])

        # We populate the lists
        for i in range(len(expression)):
            op_type, mode_i = expression[i]

            # First, we check whether the whole string vanishes by nilpotent substring
            if op_type == cur_op_type[mode_i]:
                return(cls.zero())
            cur_op_type[mode_i] = op_type
            if first_op_type[mode_i] is None:
                first_op_type[mode_i] = op_type


            if op_type == 'c':
                C[mode_i].append(sum(N[mode_i: ]))
            elif op_type == 'a':
                A[mode_i].append(sum(N[mode_i: ]))

            N[mode_i] += 1

        # Now we specify each substring's signature by checking the first and last op type.
        # We also calculate S-numbers. S[mode index m] = total number of operators with mode index m or higher.
        # If mode is missing from the expression, the signature is set to None. We explicitly skip the associated mode properties in further calculations.
        S = [0] * M
        for i in range(M):
            if first_op_type[i] is None:
                signature[i] = None
            else:
                signature[i] = first_op_type[i] + cur_op_type[i]
                S[i] = sum(N[i:])

        # The major parity contribution, which is equal amongst all terms:
        major_parity = 0
        for i in range(M):
            if signature[i] is None:
                continue
            major_parity += sum(C[i]) + sum(A[i])

        # Now, we go mode index by mode index and adjust the parity based on the signature. We also keep track of the remaining operators
        fixed_C_op = set()
        fixed_A_op = set()
        AC_modes = [] # list of mode indices with the A/C signature. These double the number of terms (0-pair term and 1-pair term)
        for i in range(M):
            if signature[i] is None:
                continue
            if signature[i] == 'cc':
                major_parity += len(A[i])
                fixed_C_op.add(i)
            elif signature[i] == 'aa':
                major_parity += S[i] + len(C[i]) + 1
                fixed_A_op.add(i)
            elif signature[i] == 'ca':
                major_parity += S[i] + len(C[i])
                fixed_C_op.add(i)
                fixed_A_op.add(i)
            elif signature[i] == 'ac':
                major_parity += len(C[i])
                AC_modes.append(i)

        # Now we finally put together our terms
        terms = {}
        for term_inclusion in binit(len(AC_modes), invert = True):
            cur_C_op = set(fixed_C_op)
            cur_A_op = set(fixed_A_op)
            cur_parity = major_parity
            for i in range(len(AC_modes)):
                if term_inclusion[i]:
                    cur_C_op.add(AC_modes[i])
                    cur_A_op.add(AC_modes[i])
                    cur_parity += S[AC_modes[i]] + 1
            terms[(frozenset(cur_C_op), frozenset(cur_A_op))] = sign(cur_parity) * coef

        return(cls(terms))






    @classmethod
    def T(cls, i, j):
        # Creates a simple transition operator
        return(cls((i, j)))


    def __init__(self, init_object):
        # The object represents its expression as a dictionary whose keys are
        # tuples ({c}, {a}), which are the sets of mode indices with creation
        # and annihilation operators, respectively, indexed from 0, and whose
        # values are the corresponding coefficients. It is understood that keys
        # not present in the dictionary have coefficient 0.

        # The object can be initialised in multiple ways:
        #   -by providing the dictionary directly
        #   -by providing a list or set of tuples in one of these forms:
        #      a) ({c}, {a}, coef)
        #      b) ({c}, {a}), with the coef being implicitly understood to be 1
        #    These can be mixed in the provided iterable.
        #    {c}, {a} can also be provided as integers, in which case they are
        #    understood to be single-elements sets.
        #   -by providing a single tuple in one of the forms above. The object
        #    is understood to only have a single term.
        #   -by providing a single number, in which the object is understood to
        #    be a scalar.
        #   -by providing a list of ndarrays. Each ndarray must be of the
        #    dimension M^(2.n) and is understood to contribute a linear comb.
        #    of terms with n creation and n annihilation operators.
        #      a) For an (M, M) ndarray, the correspondence is
        #           A[i][j] = coef of c = {i}, a = {j}
        #      b) For an (M, M, M, M) ndarray, the correspondence is
        #           A[i][j][k][l] = coef of c = {i, j}, a = {k, l}
        #    The required symmetry in the object is not checked to save time.

        # In an abstract sense, the object is equivalent to a map
        #   phi : P(N_0) x P(N_0) -> C
        # where all but a finite elements of the domain are mapped to 0.

        if isinstance(init_object, dict):
            # We validate the dict. We only check the shape of the iterable parts of the keys, not whether all elements are integers
            for key in init_object:
                if isinstance(key, tuple):
                    if len(key) == 2:
                        c, a = key
                        if not isinstance(c, frozenset):
                            raise Exception(f"Operator initialised from dict with invalid keys: first element not a 'frozenset' for key {key}")
                        if not isinstance(a, frozenset):
                            raise Exception(f"Operator initialised from dict with invalid keys: second element not a 'frozenset' for key {key}")
                        if not is_number(init_object[key]):
                            raise Exception(f"Operator initialised from dict with invalid values: key {key} assigned a non-numerical value of type '{type(init_object[key])}'")
                    else:
                        raise Exception(f"Operator initialised from dict with invalid keys: lenght != 2 for key {key}")
                else:
                    raise Exception(f"Operator initialised from dict with invalid keys: type != 'tuple' for key {key}")

            self.phi = init_object
        elif isinstance(init_object, list) or isinstance(init_object, set):
            self.phi = {}
            for term in init_object:
                key, val = self.parse_tuple_atom(term)
                if key not in self.phi:
                    self.phi[key] = val
                else:
                    # Redundant terms in init_object are forbidden
                    raise Exception(f"Multiple terms provided which correspond to the same operator string. Offending operator string: {self.term_to_str(key)}")
        elif isinstance(init_object, tuple):
            key, val = self.parse_tuple_atom(init_object)
            self.phi = {key : val}
        elif is_number(init_object):
            self.phi = {(frozenset(), frozenset()) : init_object}
        elif isinstance(init_object, np.ndarray):
            raise NotImplementedError("ndarray initialisation not yet implemented")
            #TODO
        else:
            raise Exception(f"Provided elements must be one of the following types: 'dict', 'list', 'set', 'tuple', 'ndarray', or a scalar; provided type = '{type(init_object)}'")

        self.simplify()



    def parse_tuple_atom(self, tup):
        # Output is ({c}, {a}, coef)
        # If coef is not in the input, it is assumed to be 1
        # If c, a are int, we wrap them into sets to standardise the expression

        if isinstance(tup, tuple):

            if len(tup) == 2:
                cur_coef = 1.0
            elif len(tup) == 3:
                cur_coef = tup[2]
            else:
                raise Exception(f"Provided singular operator element must be of length 2 or 3; provided length = {len(tup)}")

            if isinstance(tup[0], set):
                cur_c = frozenset(tup[0])
            elif isinstance(tup[0], int):
                cur_c = frozenset({tup[0]})
            elif isinstance(tup[0], frozenset):
                cur_c = tup[0]
            else:
                raise Exception(f"Provided creation operator sequence must be of type 'set' or 'int' (for a single creation operator); provided type = '{type(tup[0])}'")

            if isinstance(tup[1], set):
                cur_a = frozenset(tup[1])
            elif isinstance(tup[1], int):
                cur_a = frozenset({tup[1]})
            elif isinstance(tup[1], frozenset):
                cur_a = tup[1]
            else:
                raise Exception(f"Provided annihilation operator sequence must be of type 'set', 'frozenset', or 'int' (for a single annihilation operator); provided type = '{type(tup[1])}'")

            return(((cur_c, cur_a), cur_coef))

        elif is_number(tup):
            # We assume tup is just the number and represents a scalar term.
            return(((frozenset(), frozenset()), tup))
        else:
            raise Exception(f"Provided term must be of type 'tuple' or any numerical type; provided type = '{type(tup)}'")

    def simplify(self):
        # This method mutates self by deleting all phi entries whose value is zero
        for key in list(self.phi.keys()):
            if self.phi[key] == 0.0:
                del self.phi[key]

    # ------------------------------ Descriptors ------------------------------

    def __str__(self):
        # Prints the entire expression, shortening c_i to i^ and a_i to just i
        if len(self.phi) == 0:
            return("0")
        out = ""
        is_first_term = True
        for key, val in self.phi.items():
            out += self.term_to_str(key, is_first_term)
            is_first_term = False
        return(out)

    def __repr__(self):
        return(f"<Operator.Operator object; {str(self)}>")

    def term_to_str(self, key, omit_trivial_prefix = True):
        # term is a tuple ({c}, {a}, {coef})
        c, a = key
        if key in self.phi:
            coef = self.phi[key]
        else:
            # zero
            return("")

        is_scalar = (len(c) == 0 and len(a) == 0)

        mult_str = ""
        if not is_scalar:
            mult_str = " . "

        # coef prefix
        prefix = ""

        # We take care of the zero case
        if coef == 0:
            return("")

        # We take care of the nontrivial complex case
        if has_imag_comp(coef):
            if has_real_comp(coef):
                if coef.real > 0:
                    if coef.imag > 0:
                        if omit_trivial_prefix:
                            prefix = f"({coef.real} + {coef.imag}i)" + mult_str
                        else:
                            prefix = f" + ({coef.real} + {coef.imag}i)" + mult_str
                    else:
                        if omit_trivial_prefix:
                            prefix = f"({coef.real} - {abs(coef.imag)}i)" + mult_str
                        else:
                            prefix = f" + ({coef.real} - {abs(coef.imag)}i)" + mult_str
                else:
                    if coef.imag > 0:
                        if omit_trivial_prefix:
                            prefix = f"({coef.imag}i - {abs(coef.real)})" + mult_str
                        else:
                            prefix = f" + ({coef.imag}i - {abs(coef.real)})" + mult_str
                    else:
                        if omit_trivial_prefix:
                            prefix = f"- ({abs(coef.real)} + {abs(coef.imag)}i)" + mult_str
                        else:
                            prefix = f" - ({abs(coef.real)} + {abs(coef.imag)}i)" + mult_str
            else:
                if coef.imag > 0:
                    if omit_trivial_prefix:
                        prefix = f"{coef.imag}i" + mult_str
                    else:
                        prefix = f" + {coef.imag}i" + mult_str
                else:
                    if omit_trivial_prefix:
                        prefix = f"- {abs(coef.imag)}i" + mult_str
                    else:
                        prefix = f" - {abs(coef.imag)}i" + mult_str
        elif isinstance(coef, (complex, np.complexfloating)):
            # Complex type but no imaginary part
            coef = coef.real

        # We take care of the real case
        if coef == 1:
            if omit_trivial_prefix:
                if is_scalar:
                    prefix = "1"
                else:
                    prefix = ""
            else:
                if is_scalar:
                    prefix = " + 1"
                else:
                    prefix = " + "
        elif coef == -1:
            if omit_trivial_prefix:
                if is_scalar:
                    prefix = "- 1"
                else:
                    prefix = "- "
            else:
                if is_scalar:
                    prefix = " - 1"
                else:
                    prefix = " - "
        elif coef > 0:
            if omit_trivial_prefix:
                prefix = f"{coef}" + mult_str
            else:
                prefix = f" + {coef}" + mult_str
        elif coef < 0:
            if omit_trivial_prefix:
                prefix = f"- {np.abs(coef)}" + mult_str
            else:
                prefix = f" - {np.abs(coef)}" + mult_str

        if is_scalar:
            return(prefix)
        # main term
        main_str = ""
        for c_op in sorted(c):
            main_str += f"{c_op}^ "
        for a_op in sorted(a, reverse = True):
            main_str += f"{a_op} "
        return(prefix + main_str[:-1])

    # --------------- Scalar multiplication, addition, products ---------------

    # Self-mutating operations

    def scale_self_by_scalar(self, scalar):
        if scalar == 0.0:
            self.phi = {}
        else:
            for key in self.phi:
                self.phi[key] *= scalar

    def add_to_self(self, other):
        for key in other.phi:
            if key in self.phi:
                self.phi[key] += other.phi[key]
            else:
                self.phi[key] = other.phi[key]
        self.simplify()

    # Object-creating operations

    def scalar_multiplication(self, scalar):
        res = Operator.copy(self)
        res.scale_self_by_scalar(scalar)
        return(res)

    def addition(self, other):
        res = Operator.copy(self)
        res.add_to_self(other)
        return(res)

    def product(self, other):
        res = Operator.zero()

        for key_i in self.phi:
            for key_j in other.phi:
                cur_coef = self.phi[key_i] * other.phi[key_j]
                cur_expression = []
                c_i, a_i = key_i
                c_j, a_j = key_j
                for c_i_op in sorted(c_i):
                    cur_expression.append(('c', c_i_op))
                for a_i_op in sorted(a_i, reverse = True):
                    cur_expression.append(('a', a_i_op))
                for c_j_op in sorted(c_j):
                    cur_expression.append(('c', c_j_op))
                for a_j_op in sorted(a_j, reverse = True):
                    cur_expression.append(('a', a_j_op))

                res.add_to_self(Operator.normal_order(cur_expression, cur_coef))
        return(res)

    def __add__(self, other):
        if isinstance(other, Operator):
            return(self.addition(other))
        elif is_number(other):
            return(self.addition(Operator(other)))
        else:
            raise Exception(f"Only Operators and numbers can be added to an Operator; provided type is '{type(other)}'")

    def __radd__(self, other):
        return(self.__add__(other))

    def __sub__(self, other):
        if isinstance(other, Operator):
            res = other.scalar_multiplication(-1)
            res.add_to_self(self)
            return(res)
        elif is_number(other):
            return(self.addition(Operator( (-1.0) * other)))
        else:
            raise Exception(f"Only Operators and numbers can be subtracted from an Operator; provided type is '{type(other)}'")

    def __rsub__(self, other):
        return(-(self.__sub__(other)))

    def __mul__(self, other):
        if isinstance(other, Operator):
            return(self.product(other))
        elif is_number(other):
            return(self.scalar_multiplication(other))
        else:
            raise Exception(f"Operators can only by multiplied by another Operator or by a number; provided type is '{type(other)}'")

    def __rmul__(self, other):
        if is_number(other):
            return(self.scalar_multiplication(other))
        else:
            raise Exception(f"Operators can only by multiplied by another Operator or by a number; provided type is '{type(other)}'")

    def __neg__(self):
        return(self.scalar_multiplication(-1))

    # --------------------- Overlap with coherent states ----------------------

    def mel(self, psi_1, psi_2):
        # here psi_1/2 are two objects which posses the norm_overlap methods
        # (such as any CS_base or derived objects).

        # norm_overlap has the form psi_1.norm_overlap(psi_2, c, a), where c, a
        # are lists of indices on which creation and annihilation operators act
        # in normal order for the ket and inverted order on the bra.

        res = 0.0

        for ops, coef in self.phi.items():
            c, a = ops
            overlap_term = psi_1.norm_overlap(psi_2, sorted(c), sorted(a))
            res += coef * overlap_term

        return(res)

