
from class_Semaphor import Semaphor



class Journal():

    # Constructor, descriptor, output

    semaphor_prefixes = ["| ", "/ ", "- ", "\\ "]

    def __init__(self, verbosity = 0, print_on_the_fly = print):

        self.journal_text = ""
        self.routine_stack = [] # every routine is [header, max required verbosity]
        self.verbosity = verbosity # The higher the verbosity, the more in detail the Journal is
        self.print_on_the_fly = print_on_the_fly # If not None, the messages are printed with the provided function on the fly
        self.depth = 0

        self.semaphor = Semaphor(time_format = "%H:%M:%S", print_directly = False) # The in-house semaphor used to hang on a routine
        self.cur_semaphor_event = None
        self.cur_semaphor_prefix_id = 0

    def dump(self):
        # dumps journal into string
        return(self.journal_text)

    # Helping methods

    def pre(self):
        # prefix for current depth
        return("| " * self.depth)

    def get_stack_v(self):
        # If stack is non-empty, we return last max req verbosity from stack; -1 otherwise (no requirement)
        if len(self.routine_stack) > 0:
            return(self.routine_stack[-1][1])
        return(-1)

    # Write method

    def write(self, msg, v = -1):
        # Writes a message inside current routine at current depth
        if self.verbosity >= max(v, self.get_stack_v()):
            self.journal_text += self.pre() + msg + "\n"
            if self.print_on_the_fly is not None:
                self.print_on_the_fly(self.pre() + msg)

    # Routine stack management

    def enter(self, header, v = -1, semaphored = False, **kwargs):
        # header: human-readable name of routine
        # v: verbosity requirement
        # semaphored: if True, the routine is semaphored
        # if semaphored, kwargs are provided to semaphor

        self.routine_stack.append([header, max(v, self.get_stack_v())])

        if semaphored:
            if self.verbosity >= max(v, self.get_stack_v()):
                s_nl = False
                if "newline" in kwargs:
                    s_nl = kwargs["newline"]
                semaphor_event_ID, semaphor_header = self.semaphor.create_event(kwargs["tau_space"], header, s_nl)
                self.cur_semaphor_event = semaphor_event_ID
                self.write(semaphor_header, v)

        else:
            self.write(header, v)
        self.depth += 1

    def update_semaphor_event(self, tau):
        if self.cur_semaphor_event is not None:
            msg = self.semaphor.update(self.cur_semaphor_event, tau)
            if msg is not None:
                # We don't write update messages into the log which meant to be saved, only to print on the fly
                self.print_on_the_fly("| " * (self.depth - 1) + Journal.semaphor_prefixes[self.cur_semaphor_prefix_id] + msg, end='\r')
                self.cur_semaphor_prefix_id = (self.cur_semaphor_prefix_id + 1) % len(Journal.semaphor_prefixes)

    def exit(self, end_message = None):
        last_routine = self.routine_stack.pop()


        if end_message is not None:
            exit_msg = " " + end_message
        else:
            exit_msg = ""

        # check if it's semaphored
        if self.cur_semaphor_event is not None:
            # We exit the semaphored routine
            process_duration, semaphor_exit_msg = self.semaphor.finish_event(self.cur_semaphor_event, end_message)
            self.cur_semaphor_event = None
            self.cur_semaphor_prefix_id = 0
            exit_msg = " " + semaphor_exit_msg
        self.depth -= 1
        self.write(f"\_{exit_msg}", last_routine[1])


