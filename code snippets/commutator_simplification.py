
# This code simplifies the expression [ A, B ], where A and B are general products of raising and lowering operators

def contraction_specific_indices(list_of_indices, number_of_contractions):
    if number_of_contractions == 0:
        return([[]]) # Returns one contraction, which has no contraction pairs (the empty contraction)
    # The added contraction lower-higher, to avoid duplicates
    res = []
    for lower_paired_element in range(len(list_of_indices)-1-(number_of_contractions - 1)*2):
        for higher_paired_element in range(lower_paired_element+1, len(list_of_indices)):
            cur_prefix = [list_of_indices[lower_paired_element], list_of_indices[higher_paired_element]]
            cur_reduced_indices = list_of_indices.copy()
            del cur_reduced_indices[higher_paired_element]
            del cur_reduced_indices[0:lower_paired_element+1]
            reduced_contractions = contraction_specific_indices(cur_reduced_indices, number_of_contractions - 1)
            for i in range(len(reduced_contractions)):
                res.append([cur_prefix] + reduced_contractions[i])
    return(res)

def contraction_indices(number_of_elements, number_of_contractions):
    # Returns [contraction configuration][contraction pairing index] = [first index, second index]
    specific_indices = []
    for i in range(number_of_elements):
        specific_indices.append(i)
    return(contraction_specific_indices(specific_indices, number_of_contractions))

def optimized_contraction_indices(lowering_indices, raising_indices):
    # Optimized algorithm
    # lowering/raising_indices are arrays of indices of L/R ladder operators in the given sequence, in ascending order
    if len(raising_indices) == 0 or len(lowering_indices) == 0:
        return([[]]) # empty contraction is the only option
    if lowering_indices[0] > raising_indices[-1]:
        return([[]]) # already normal-ordered
    if len(lowering_indices) == 1:
        # This is the last one
        res = [[]]
        for raising_index in raising_indices:
            if raising_index > lowering_indices[0]:
                res.append([[lowering_indices[0], raising_index]])
        return(res)
    # Two options: either the first L is contracted or it isn't
    res = optimized_contraction_indices(lowering_indices[1:], raising_indices)
    for i in range(len(raising_indices)):
        if raising_indices[i] > lowering_indices[0]:
            cur_contractions = optimized_contraction_indices(lowering_indices[1:], raising_indices[:i] + raising_indices[i+1:])
            for cur_contraction in cur_contractions:
                res.append([[lowering_indices[0], raising_indices[i]]] + cur_contraction)
    return(res)

class Operator():
    def __init__(self, operator_type, operator_indices):
        self.t = operator_type # "L/R/K" for lowering/raising/kronecker
        self.i = operator_indices # "index" for L/R and ["index", "index"] for K
        if self.t in ["L", "R"]:
            self.cls = "ladder"
        elif self.t == "K":
            self.cls = "kronecker"
        elif self.t == "scalar":
            self.cls = "scalar"

    def commutator(self, other):
        # other is also an instance of Operator
        # returns a tuple (sign, Operator)
        if self.t == "K" or other.t == "K":
            return("zero", None)
        if self.t == other.t:
            return("zero", None)
        if self.t == "L" and other.t == "R":
            if self.i == other.i:
                return("plus", Operator("scalar", 1))
            else:
                return("plus", Operator("K", [self.i, other.i]))
        if self.t == "R" and other.t == "L":
            if self.i == other.i:
                return("minus", Operator("scalar", 1))
            else:
                return("minus", Operator("K", [self.i, other.i]))
        return(f"ERROR: Unknown types ({self.t}, {other.t})")

    def contraction(self, other):
        # Returns the contraction self.other - :self.other:
        if self.t == "K" or other.t == "K":
            return("zero", None)
        if self.t == other.t:
            return("zero", None)
        if self.t == "R" and other.t == "L":
            return("zero", None)
        if self.t == "L" and other.t == "R":
            if self.i == other.i:
                return("plus", Operator("scalar", 1))
            else:
                return("plus", Operator("K", [self.i, other.i]))
        return(f"ERROR: Unknown types ({self.t}, {other.t})")

    def to_string(self):
        if self.cls == "ladder":
            return(f"{self.t}{self.i}")
        elif self.cls == "kronecker":
            return(f"{{{self.i[0]},{self.i[1]}}}")

class Expression():

    def user_input_expression(msg):
        while(True):
            candidate = input(msg)
            candidate_expression = decode_expression(candidate)
            if candidate_expression is not None:
                return(candidate_expression)

    def __init__(self, operator_sequence):
        self.operator_sequence = operator_sequence # [n] = {Operator()}
        self.distinct_indices = []
        for operator in self.operator_sequence:
            if operator.cls == "ladder":
                if operator.i not in self.distinct_indices:
                    self.distinct_indices.append(operator.i)
            elif operator.cls == "kronecker":
                for ind in operator.i:
                    if ind not in self.distinct_indices:
                        self.distinct_indices.append(ind)
        self.prefactor = 1

    def get_normal_ordered_signature(self):
        # The signature is a set, where each element is a set of equal indices and then two special signature strings "L{number}" and "R{number}", which count the numebr of raising and lowering operators for that set of equal indices. This signature is only useful if the Expression is normal ordered.
        equal_index_sets = []
        for ind in self.distinct_indices:
            equal_index_sets.append([ind])
        for i in range(len(self.operator_sequence)):
            if self.operator_sequence[i].cls == "kronecker":
                # If the two indices are both in the same equal index set already, the operator is obsolete
                # If the two indices are in two existing equal index sets, the two sets merge
                i_one = None
                i_two = None
                for j in range(len(equal_index_sets)):
                    if self.operator_sequence[i].i[0] in equal_index_sets[j]:
                        i_one = j
                    if self.operator_sequence[i].i[1] in equal_index_sets[j]:
                        i_two = j
                if i_one is None and i_two is None:
                    equal_index_sets.append([self.operator_sequence[i].i[0], self.operator_sequence[i].i[1]])
                elif i_one is None:
                    equal_index_sets[i_two].append(self.operator_sequence[i].i[0])
                elif i_two is None:
                    equal_index_sets[i_one].append(self.operator_sequence[i].i[1])
                else:
                    # Both are non-zero
                    if i_one != i_two:
                        for j in range(len(equal_index_sets[i_two])):
                            equal_index_sets[i_one].append(equal_index_sets[i_two][j])
                        del equal_index_sets[i_two]
        # Convert all to sets
        res_equal_index_sets = set()
        for i in range(len(equal_index_sets)):
            # For each equal index set, we now count the number of raising and lowering operators
            L_number = 0
            R_number = 0
            for op in self.operator_sequence:
                if op.t == "L":
                    if op.i in equal_index_sets[i]:
                        L_number += 1
                elif op.t == "R":
                    if op.i in equal_index_sets[i]:
                        R_number += 1
            res_equal_index_sets.add(frozenset(equal_index_sets[i]))
            res_equal_index_sets.add(frozenset([f"L{L_number}", f"R{R_number}"]))
        return(res_equal_index_sets)

    def normal_ordered_expression(self):
        # Returns an expression with DIFFERENT VALUE, which constitues the same operators, except normal ordered
        self.simplify_expression()
        # Now there are no scalar operators
        raising_operators = []
        lowering_operators = []
        kronecker_operators = []
        for i in range(len(self.operator_sequence)):
            if self.operator_sequence[i].t == "R":
                raising_operators.append(self.operator_sequence[i])
            elif self.operator_sequence[i].t == "L":
                lowering_operators.append(self.operator_sequence[i])
            elif self.operator_sequence[i].t == "K":
                kronecker_operators.append(self.operator_sequence[i])
        res = Expression(raising_operators + lowering_operators + kronecker_operators)
        res.prefactor = self.prefactor
        return(res)

    def simplify_expression(self):
        # Resolves signs, removes scalars, simplifies kronecker deltas

        # Scalar removal
        ladder_operator_sequence = []
        kronecker_operator_sequence = []
        equal_index_sets = [] # each element is a list of indices which are all set equal to each other via kronecker pairings
        for i in range(len(self.operator_sequence)):
            if self.operator_sequence[i].t == "scalar":
                self.prefactor *= self.operator_sequence[i].i
            elif self.operator_sequence[i].cls == "ladder":
                ladder_operator_sequence.append(self.operator_sequence[i])
            elif self.operator_sequence[i].cls == "kronecker":
                # If the two indices are both in the same equal index set already, the operator is obsolete
                # If the two indices are in two existing equal index sets, the two sets merge
                i_one = None
                i_two = None
                is_obsolete = False
                for j in range(len(equal_index_sets)):
                    if self.operator_sequence[i].i[0] in equal_index_sets[j]:
                        i_one = j
                    if self.operator_sequence[i].i[1] in equal_index_sets[j]:
                        i_two = j
                if i_one is None and i_two is None:
                    equal_index_sets.append([self.operator_sequence[i].i[0], self.operator_sequence[i].i[1]])
                elif i_one is None:
                    equal_index_sets[i_two].append(self.operator_sequence[i].i[0])
                elif i_two is None:
                    equal_index_sets[i_one].append(self.operator_sequence[i].i[1])
                else:
                    # Both are non-zero
                    if i_one == i_two:
                        is_obsolete = True
                    else:
                        for j in range(len(equal_index_sets[i_two])):
                            equal_index_sets[i_one].append(equal_index_sets[i_two][j])
                        del equal_index_sets[i_two]
                if not is_obsolete:
                    kronecker_operator_sequence.append(self.operator_sequence[i])
        self.operator_sequence = ladder_operator_sequence + kronecker_operator_sequence


    def to_string(self):
        self.simplify_expression()
        res_list = []
        if self.prefactor != 1:
            res_list.append(str(self.prefactor))
        for operator in self.operator_sequence:
            """if operator.cls == "ladder":
                res_list.append(f"{operator.t}{operator.i}")
            elif operator.cls == "kronecker":
                res_list.append(f"{{{operator.i[0]},{operator.i[1]}}}")"""
            res_list.append(operator.to_string())
        if len(res_list) == 0:
            # Just one
            return("1")
        return(".".join(res_list))

    def is_equal(self, other):
        # Determines if is equal to another Expression up to prefactor.
        # NOTE: Assumes both Expressions are normal-ordered
        # This is done by comparing all the Ls and Rs and then all equal_index_sets
        return(self.get_normal_ordered_signature() == other.get_normal_ordered_signature())


    def commutator(self, other):
        # other is also an instance of Expression
        # Returns an Expression_sum
        res_expression_sum = Expression_sum([])
        for i in range(len(self.operator_sequence)):
            prefix_sequence = self.operator_sequence[:i]
            suffix_sequence = self.operator_sequence[i+1:]
            # res is prefix_sequence . [this operator, other expression] . suffix_sequence
            cur_expression_sum = Expression_sum([])
            for j in range(len(other.operator_sequence)):
                other_prefix_sequence = other.operator_sequence[:j]
                other_suffix_sequence = other.operator_sequence[j+1:]
                commutator_sign, commutator_operator = self.operator_sequence[i].commutator(other.operator_sequence[j])
                if commutator_sign == "zero":
                    # This term disappears
                    continue
                cur_expression = Expression(prefix_sequence + other_prefix_sequence + [commutator_operator] + other_suffix_sequence + suffix_sequence)
                if commutator_sign == "plus":
                    cur_expression_sum.list_of_expressions.append(cur_expression)
                elif commutator_sign == "minus":
                    cur_expression.prefactor *= -1
                    cur_expression_sum.list_of_expressions.append(cur_expression)
            res_expression_sum = res_expression_sum + cur_expression_sum
        return(res_expression_sum)

    def nonzero_contractions(self):
        # For a nonzero contraction, each pair must be between L from the left and R on the right
        contractions = []
        lowering_indices = []
        raising_indices = []
        for i in range(len(self.operator_sequence)):
            if self.operator_sequence[i].t == "L":
                lowering_indices.append(i)
            elif self.operator_sequence[i].t == "R":
                raising_indices.append(i)
        return(optimized_contraction_indices(lowering_indices, raising_indices))




class Expression_sum():

    def normal_ordering(unordered_expression):
        # Takes an expression and returns an Expression_sum which constitues only normal-ordered expressions, and is equal to the original expression
        unordered_expression.simplify_expression()
        res_expression_sum = Expression_sum([])
        for contraction in unordered_expression.nonzero_contractions():
            uncontracted_indices = []
            for i in range(len(unordered_expression.operator_sequence)):
                uncontracted_indices.append(i)
            for pair in contraction:
                if pair[0] in uncontracted_indices:
                    uncontracted_indices.remove(pair[0])
                if pair[1] in uncontracted_indices:
                    uncontracted_indices.remove(pair[1])
            new_operator_sequence = []
            # First, we add the contracted operators
            is_term_destroyed = False
            for pair in contraction:
                pair_sign, pair_operator = unordered_expression.operator_sequence[pair[0]].contraction(unordered_expression.operator_sequence[pair[1]])
                if pair_sign == "zero":
                    is_term_destroyed = True
                    break
                elif pair_sign == "plus":
                    new_operator_sequence.append(pair_operator)
            if not is_term_destroyed:
                for uncontracted_i in uncontracted_indices:
                    new_operator_sequence.append(unordered_expression.operator_sequence[uncontracted_i])
                res_expression_sum.list_of_expressions.append(Expression(new_operator_sequence).normal_ordered_expression())
        """number_of_contractions = 0
        res_expression_sum = Expression_sum([])
        while(number_of_contractions * 2 <= len(unordered_expression.operator_sequence)):
            cur_contractions = contraction_indices(len(unordered_expression.operator_sequence), number_of_contractions)
            for contraction in cur_contractions:
                uncontracted_indices = []
                for i in range(len(unordered_expression.operator_sequence)):
                    uncontracted_indices.append(i)
                for pair in contraction:
                    if pair[0] in uncontracted_indices:
                        uncontracted_indices.remove(pair[0])
                    if pair[1] in uncontracted_indices:
                        uncontracted_indices.remove(pair[1])
                new_operator_sequence = []
                # First, we add the contracted operators
                is_term_destroyed = False
                for pair in contraction:
                    pair_sign, pair_operator = unordered_expression.operator_sequence[pair[0]].contraction(unordered_expression.operator_sequence[pair[1]])
                    if pair_sign == "zero":
                        is_term_destroyed = True
                        break
                    elif pair_sign == "plus":
                        new_operator_sequence.append(pair_operator)
                if not is_term_destroyed:
                    for uncontracted_i in uncontracted_indices:
                        new_operator_sequence.append(unordered_expression.operator_sequence[uncontracted_i])
                    res_expression_sum.list_of_expressions.append(Expression(new_operator_sequence).normal_ordered_expression())

            number_of_contractions += 1"""
        res_expression_sum.join_equal_terms()
        # We multiply every constituent expression by the prefactor of the original expression
        for i in range(len(res_expression_sum.list_of_expressions)):
            res_expression_sum.list_of_expressions[i].prefactor *= unordered_expression.prefactor
        return(res_expression_sum)

    def __init__(self, list_of_expressions):
        self.list_of_expressions = list_of_expressions

    def to_string(self):
        self.join_equal_terms()
        res = self.list_of_expressions[0].to_string()
        for i in range(1, len(self.list_of_expressions)):
            self.list_of_expressions[i].simplify_expression()
            if self.list_of_expressions[i].prefactor < 0:
                # There's already a leading sign of minus
                res += self.list_of_expressions[i].to_string()
            else:
                res += "+" + self.list_of_expressions[i].to_string()
        return(res)

    def normal_order_self(self):
        # Normal orders every constituent expression and puts them all together
        res_list_of_expressions = []
        for expr in self.list_of_expressions:
            normal_ordered_sum = Expression_sum.normal_ordering(expr)
            res_list_of_expressions += normal_ordered_sum.list_of_expressions
        self.list_of_expressions = res_list_of_expressions

    def join_equal_terms(self):
        list_of_distinct_expressions = []
        for expr in self.list_of_expressions:
            is_distinct = True
            for i in range(len(list_of_distinct_expressions)):
                if expr.is_equal(list_of_distinct_expressions[i]):
                    list_of_distinct_expressions[i].prefactor += expr.prefactor
                    is_distinct = False
                    break
            if is_distinct:
                list_of_distinct_expressions.append(expr)
        self.list_of_expressions = list_of_distinct_expressions

    def __add__(self, other):
        return(Expression_sum(self.list_of_expressions + other.list_of_expressions))

def decode_expression(expression_string):
    # Output is an instance of Expression()
    try:
        list_of_operator_strings = expression_string.split(".")
        operator_sequence = []
        for i in range(len(list_of_operator_strings)):
            if list_of_operator_strings[i][0] not in ["L", "R", "{"]:
                raise Exception(f"Unrecognised operator type for {list_of_operator_strings[i]}")
            if list_of_operator_strings[i][0] in ["L", "R"]:
                cur_type = list_of_operator_strings[i][0]
                cur_value = list_of_operator_strings[i][1:]
            elif list_of_operator_strings[i][0] == "{":
                cur_type = "K"
                cur_value = list_of_operator_strings[i].lstrip("{").rstrip("}").split(",")
            operator_sequence.append(Operator(cur_type, cur_value))
        return(Expression(operator_sequence))
    except Exception as e:
        print("Try again;", e)
        return(None)


"""print("Syntax: a_i.a^dag_j is encoded as \"Li.Rj\", i.e. every operator is either R(aising) or L(owering). The Kronecker delta is represented as {i,j} in the output.")
print("The only rules for simplification are: [Li, Lj] = [Ri, Rj] = 0, [Li, Rj] = {i, j}, and [A, B.C] = [A, B].C + B.[A, C].")

expression_A = Expression.user_input_expression("Input the expression on the left:")
expression_B = Expression.user_input_expression("Input the expression on the right:")

commutator_result = expression_A.commutator(expression_B)
commutator_result.normal_order_self()

print(commutator_result.to_string())"""

expression_A = Expression.user_input_expression("Input the expression on the left:")
print("Nonequivalent normal ordered:", expression_A.normal_ordered_expression().to_string())

normal_ordered_A = Expression_sum.normal_ordering(expression_A)
print("Wick's theorem:", normal_ordered_A.to_string())

