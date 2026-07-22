from utils.class_Semaphor import Semaphor

# -----------------------------------------------------------------------------
# -------------------------- Typographical constants --------------------------
# -----------------------------------------------------------------------------

# -------------------------------- Delimiters ---------------------------------
nested_semaphor_delim = " | "

# ---------------------------------- Colors -----------------------------------
# ANSI escape codes (SGR parameters)

class color:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NOTUNDERLINE = '\033[24m'
    LIGHT = '\033[22m' # removes BOLD without resetting color
    DISABLE = '\033[02m'
    END = '\033[0m'

    class bg:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        ORANGE = '\033[43m'
        BLUE = '\033[44m'
        PURPLE = '\033[45m'
        CYAN = '\033[46m'
        GREY = '\033[47m'
        DEFAULT = '\033[49m'


gradual_fg_color = [ # used for pipes. Highlights routine depth. Cycles.
#    color.BLUE,
    color.BRIGHT_BLUE,
#    color.MAGENTA,
    color.BRIGHT_MAGENTA,
    color.CYAN,
    color.BRIGHT_CYAN
    ]


"""More color options
reset = '\033[0m'
bold = '\033[01m'
disable = '\033[02m'
underline = '\033[04m'
reverse = '\033[07m'
 strikethrough = '\033[09m'
  invisible = '\033[08m'

   class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'
"""

def st(a, w):
    if len(str(a)) >= w:
        return(str(a)[:w]) # we trim
    else:
        diff = w - len(str(a))
        return(" " * int(diff // 2) + str(a) + " " * int(diff - diff // 2))

def _dtstr(seconds, max_depth = 2):
    # Dynamically chooses the right format
    # max_depth is the number of different measurements (e.g. max_depth = 2: "2 days 5 hours")
    if seconds >= 60 * 60 * 24:
        # Days
        if max_depth == 1:
            return(f"{int(round(seconds / (60 * 60 * 24)))} days")
        remainder = seconds % (60 * 60 * 24)
        days = int((seconds - remainder) / (60 * 60 * 24))
        return(f"{days} days {_dtstr(remainder, max_depth - 1)}")
    if seconds >= 60 * 60:
        # Hours
        if max_depth == 1:
            return(f"{int(round(seconds / (60 * 60)))} hours")
        remainder = seconds % (60 * 60)
        hours = int((seconds - remainder) / (60 * 60))
        return(f"{hours} hours {_dtstr(remainder, max_depth - 1)}")
    if seconds >= 60:
        # Minutes
        if max_depth == 1:
            return(f"{int(round(seconds / 60))} min")
        remainder = seconds % (60)
        minutes = int((seconds - remainder) / (60))
        return(f"{minutes} min {_dtstr(remainder, max_depth - 1)}")
    if seconds >= 1:
        # Seconds
        if max_depth == 1:
            return(f"{int(round(seconds))} sec")
        remainder = seconds % (1)
        secs = int((seconds - remainder))
        return(f"{secs} sec {_dtstr(remainder, max_depth - 1)}")
    # Milliseconds
    return(f"{int(round(seconds / 0.001))} ms")

# -----------------------------------------------------------------------------
# ------------------------------- class Journal -------------------------------
# -----------------------------------------------------------------------------

class Journal():

    # Constructor, descriptor, output

    semaphor_prefixes = ["| ", "/ ", "- ", "\\ "]

    answer_strings = {
        "yes" : ["yes", "y", "aye"],
        "no" : ["no", "n"]
        }

    valid_answers = []
    for a_val in answer_strings.values():
        valid_answers += a_val

    @classmethod
    def sample_behaviour(cls, **kwargs):
        # useful when testing whether behaviour is as expected
        sample_journal = cls(**kwargs)

        sample_journal.enter("Journal behaviour test commences...")
        sample_journal.write("This is a sample instance of class Journal. Please take note of when the announced function does not agree with the output.")

        sample_journal.enter("Testing special object printing methods...")
        sample_journal.write("Special objects, such as item lists, tables, or matrices, can be printed with special formatting using these methods.")

        sample_journal.enter("Testing item list printing...")
        sample_journal.print_itemize({
            "print_itemize" : "Prints an itemized/enumerated list from a standard object. Supports nested lists.",
            "nested numbered list" : ["First element", "Second element", {"this" : "First part of third element", "that" : "Second part of third element"}, "Fourth element"],
            "nested itemized list" : {"Keyword" : "Value", "Other keyword" : "Other value"}
            })
        sample_journal.exit()

        sample_journal.enter("Testing matrix printing...")
        sample_journal.print_matrix(
            [[0.5 + 1j] * 5, [0.554 + 0.518j] * 5, [0.245] * 5, [0.5 - 0.5j] * 5, [0.5 - 1j] * 5],
            "Example matrix",
            2,
            4
            )
        sample_journal.exit()

        sample_journal.enter("Testing table printing...")
        sample_journal.print_table(
            table_name = "Current properties",
            column_names = ["value", "default", "description"],
            row_names = ["verbosity", "print_on_the_fly", "fancy_printing", "max_row_width", "yes", "plain_semaphor_N"],
            list_of_rows = [
                [sample_journal.verbosity, 5, "Only logs with lower or equal value of verbosity are printed."],
                [sample_journal.print_on_the_fly, "True", "Whether the log is also printed to an output stream."],
                [sample_journal.fancy_printing, "True", "Formatting: font weight and color, rolling last line."],
                [sample_journal.max_row_width, 200, "Rows get trimmed if length exceeds this value."],
                [sample_journal.yes, None, "If not None, used instead of asking for input."],
                [sample_journal.plain_semaphor_N, 20, "Forced number of flags for a plain semaphor event."]
                ],
            subtable_borders=[1]
            )
        sample_journal.exit()

        sample_journal.exit()

        sample_journal.enter("Testing semaphors...")
        sample_journal.write("Semaphor is used to update the user periodically when the program spends a long time on one task.")

        sample_journal.enter("Calculating gibberish", semaphored = True, tau_space = list(range(101)))

        for a in range(10):
            for b in range(10):
                # We do something very computationally heavy here, I think

                total = 0
                for i in range(1000):
                    for j in range(1000):
                        total += ((i + a) * (j + b)) % 97

                sample_journal.update_semaphor_event(a * 10 + b)
                # Don't forget to update the semaphor with the current sim time!

        duration = sample_journal.exit("Evaluation")
        sample_journal.write(f"The calculation duration was {_dtstr(duration)}")

        sample_journal.exit()

        sample_journal.enter("Testing user input collection...")
        sample_journal.write("The journal can ask yes/no questions. This pauses the program. Set the 'yes' property to assume the answers.")
        ans = sample_journal.ask_yes_no("Is this true?")
        if ans:
            sample_journal.write("The user has decided that this is true.")
        else:
            sample_journal.write("The user has decided that this is false.")
        sample_journal.exit()

        sample_journal.exit()
        sample_journal.close_journal()

        print(sample_journal.dump())



    def __init__(self, verbosity = 5, print_on_the_fly = True, fancy_printing = True, max_row_width = 200, yes = None):

        self.journal_text = ""
        self.routine_stack = [] # every routine is [header, max required verbosity]
        self.verbosity = verbosity # The higher the verbosity, the more in detail the Journal is
        self.print_on_the_fly = print_on_the_fly # If True, printing to terminal happens concurrently
        self.depth = 0

        self.plain_semaphor_N = 20 # forced number of flags for a plain semaphor event

        self.fancy_printing = fancy_printing # if True, we use colors, highlighting etc

        self.max_row_width = max_row_width # used for clever wrapping

        self.yes = yes # If not None, used instead of asking for input

        self.last_printed_line = None # Always printed with \r until next line is staged, with different colour
        self.last_printed_depth = 0

        self.semaphor = Semaphor(time_format = "%H:%M:%S", print_directly = False) # The in-house semaphor used to hang on a routine
        self.cur_semaphor_event = None
        self.cur_semaphor_prefix_id = 0

    def close_journal(self):
        if self.print_on_the_fly:
            if self.fancy_printing:
                self.commit_last_line()
            print("-" * 30 + " Process finished " + "-" * 30)

    def dump(self):
        # dumps journal into string
        return(self.journal_text)

    # Helping methods

    def pipe_stack(self, n, allow_fancy = True):
        if self.fancy_printing and allow_fancy:
            res = ""
            for i in range(n):
                res += gradual_fg_color[i % len(gradual_fg_color)] + "|" + color.END + " "
            return(res)
        else:
            return("| " * n)

    def pre(self, allow_fancy = True):
        # prefix for current depth
        return(self.pipe_stack(self.depth, allow_fancy))

    def get_stack_v(self):
        # If stack is non-empty, we return last max req verbosity from stack; -1 otherwise (no requirement)
        if len(self.routine_stack) > 0:
            return(self.routine_stack[-1][1])
        return(-1)

    # fancy printing methods

    def cur_highlight(self, raw):
        return(color.BOLD + raw + color.LIGHT)

    def commit_last_line(self):
        if self.last_printed_line is not None:
            print(self.pipe_stack(self.last_printed_depth) + self.last_printed_line)
            self.last_printed_line = None

    def preview_print(self, msg):
        # If fancy printing is enabled, we always pre-print the last line in different colour
        print(self.pre() + self.cur_highlight(msg), end="\r")
        self.last_printed_line = msg
        self.last_printed_depth = self.depth

    def spinning_pipe(self):
        if self.fancy_printing:
            return(gradual_fg_color[(self.depth - 1) % len(gradual_fg_color)] + Journal.semaphor_prefixes[self.cur_semaphor_prefix_id] + color.END)
        else:
            return(Journal.semaphor_prefixes[self.cur_semaphor_prefix_id])

    # Write method

    def write(self, msg, v = -1, prevent_printing_on_the_fly = False, plain_msg = None):
        # Writes a message inside current routine at current depth
        if self.verbosity >= max(v, self.get_stack_v()):
            if plain_msg is None:
                self.journal_text += self.pre(False) + msg + "\n"
            else:
                self.journal_text += self.pre(False) + plain_msg + "\n"
            if self.print_on_the_fly and not prevent_printing_on_the_fly:
                if self.fancy_printing:
                    # We firstly print the last submitted line, and then the preprint of the current line
                    self.commit_last_line()
                    self.preview_print(msg)
                else:
                    # crude printing only
                    print(self.pre() + msg)

    # Special object printing methods

    def print_itemize(self, std_object):
        # std_object is an object with the following properties:
        #   1. It is either a dict or a list
        #   2. Every element is either a valid std_object or a string
        def peel_layer(cur_std_object, key = None, itemize_depth = 0):
            if isinstance(cur_std_object, dict):
                if key is not None:
                    plain_msg = " " * (2 * itemize_depth) + "-" + str(key)
                    if self.fancy_printing:
                        self.write(" " * (2 * itemize_depth) + color.BRIGHT_YELLOW + "-" + color.END + str(key), plain_msg = plain_msg)
                    else:
                        self.write(plain_msg)
                for label, value in cur_std_object.items():
                    peel_layer(value, label, itemize_depth + 1)
            elif isinstance(cur_std_object, list):
                if key is not None:
                    plain_msg = " " * (2 * itemize_depth) + "-" + str(key)
                    if self.fancy_printing:
                        self.write(" " * (2 * itemize_depth) + color.BRIGHT_YELLOW + "-" + color.END + str(key), plain_msg = plain_msg)
                    else:
                        self.write(plain_msg)
                for i in range(len(cur_std_object)):
                    peel_layer(cur_std_object[i], str(i + 1), itemize_depth + 1)
            else:
                plain_msg = " " * (2 * itemize_depth) + f"-{key}: {cur_std_object}"
                if self.fancy_printing:
                    self.write(" " * (2 * itemize_depth) + color.BRIGHT_YELLOW + "-" + color.END + f"{key}: {cur_std_object}", plain_msg = plain_msg)
                else:
                    self.write(plain_msg)

        peel_layer(std_object)
        """for label, value in dict_of_items.items():
            if self.fancy_printing:
                self.write("  " + color.BRIGHT_YELLOW + "-" + color.END + label + f": {value}")
            else:
                self.write(f"  -{label}: {value}")"""

    def print_matrix(self, m, label, dec_points = 5, max_rows = 11):

        # First, we create the label prefix
        fancy_print_queue = []
        plain_print_queue = []
        def pq_app(msg, i):
            if self.fancy_printing:
                fancy_print_queue[i] += msg
            plain_print_queue[i] += msg
        no_of_rows = min(len(m), max_rows)

        def rdec(n):
            if isinstance(n, int):
                return(f"{n}")
            if isinstance(n, complex):
                if n.imag < 0:
                    return(f"{round(n.real, dec_points)}{round(n.imag, dec_points)}j")
                else:
                    return(f"{round(n.real, dec_points)}+{round(n.imag, dec_points)}j")
            return(f"{round(n, dec_points)}")

        fancy_pre_col = []
        plain_pre_col = []
        if no_of_rows == 1:
            if self.fancy_printing:
                fancy_pre_col = [color.BRIGHT_YELLOW + "(" + color.END]
            plain_pre_col = ["("]
        else:
            if self.fancy_printing:
                fancy_pre_col = [color.BRIGHT_YELLOW + "/" + color.END]
                for i in range(no_of_rows - 2):
                    fancy_pre_col.append(color.BRIGHT_YELLOW + "|" + color.END)
                fancy_pre_col.append(color.BRIGHT_YELLOW + "\\" + color.END)
            plain_pre_col = ["/"]
            for i in range(no_of_rows - 2):
                plain_pre_col.append("|")
            plain_pre_col.append("\\")

        for i in range(no_of_rows):
            if i == no_of_rows // 2:
                if self.fancy_printing:
                    fancy_print_queue.append(f"{label} = {fancy_pre_col[i]}")
                plain_print_queue.append(f"{label} = {plain_pre_col[i]}")
            else:
                if self.fancy_printing:
                    fancy_print_queue.append(" " * (len(label) + 3) + f"{fancy_pre_col[i]}")
                plain_print_queue.append(" " * (len(label) + 3) + f"{plain_pre_col[i]}")


        # Now, we calculate how many numbers we can afford to print to respect the line length
        col_width = [0] * len(m[0])
        for i in range(len(m)):
            for j in range(len(m[0])):
                cur_len = len(rdec(m[i][j])) + 2 # the 2 is for minimal padding
                if cur_len > col_width[j]:
                    col_width[j] = cur_len

        do_rows_need_trimming = sum(col_width) > (self.max_row_width - 2 * self.depth - len(label) - 3)
        if do_rows_need_trimming:
            trim_N = 0 # number of entries we can display
            row_leeway = self.max_row_width - 2 * self.depth - len(label) - 6 - col_width[-1]
            while(sum(col_width[:trim_N]) <= row_leeway and trim_N < len(m) - 1):
                trim_N += 1
            trim_N -= 1

        if not do_rows_need_trimming:
            # We can print the entire row
            if len(m) <= max_rows:
                # we can print the entire matrix
                for i in range(len(m)):
                    for j in range(len(m[i]) - 1):
                        pq_app(st(rdec(m[i][j]), col_width[j]), i)
                    pq_app(" " + st(rdec(m[i][len(m[i]) - 1]), col_width[-1] - 2), i)
            else:
                # we trim after max_rows - 2
                for i in range(max_rows - 2):
                    for j in range(len(m[i]) - 1):
                        pq_app(st(rdec(m[i][j]), col_width[j]), i)
                    pq_app(" " + st(rdec(m[i][len(m[i]) - 1]), col_width[-1] - 2), i)
                for j in range(len(m[0]) - 1):
                    pq_app(st("...", col_width[j]), max_rows - 2)
                pq_app(" " + st("...", col_width[-1] - 2), max_rows - 2)
                for j in range(len(m[-1]) - 1):
                    pq_app(st(rdec(m[-1][j]), col_width[j]), max_rows - 1)
                pq_app(" " + st(rdec(m[-1][len(m[-1]) - 1]), col_width[-1] - 2), max_rows - 1)
        else:
            # we have to trim the number of entries per row
            if len(m) <= max_rows:
                # we can print all rows
                for i in range(len(m)):
                    for j in range(trim_N):
                        pq_app(st(rdec(m[i][j]), col_width[j]), i)
                    pq_app("... " + st(rdec(m[i][len(m[i]) - 1]), col_width[-1] - 2), i)
            else:
                # we trim rows and columns
                for i in range(max_rows - 2):
                    for j in range(trim_N):
                        pq_app(st(rdec(m[i][j]), col_width[j]), i)
                    pq_app("... " + st(rdec(m[i][len(m[i]) - 1]), col_width[-1] - 2), i)
                for j in range(trim_N):
                    pq_app(st("...", col_width[j]), max_rows - 2)
                pq_app("    " + st("...", col_width[-1] - 2), max_rows - 2)
                for j in range(trim_N):
                    pq_app(st(rdec(m[-1][j]), col_width[j]), max_rows - 1)
                pq_app("... " + st(rdec(m[-1][len(m[-1]) - 1]), col_width[-1] - 2), max_rows - 1)

        for i in range(len(plain_print_queue)):
            if self.fancy_printing:
                self.write(fancy_print_queue[i], plain_msg = plain_print_queue[i])
            else:
                self.write(plain_print_queue[i])




    def print_table(self, table_name, column_names, row_names, list_of_rows, subtable_borders = [], header_separation = 2):
        # column_names[N], row_names[M], list_of_rows[M][N]

        fancy_print_queue = [] # each line is an element
        plain_print_queue = []

        if self.fancy_printing:
            fancy_vert_sep = color.BRIGHT_YELLOW + "|" + color.END
        plain_vert_sep = "|"
        vert_sep_len = len(plain_vert_sep)

        max_len_row_names = len(str(table_name))
        for row_name in row_names:
            if len(str(row_name)) > max_len_row_names:
                max_len_row_names = len(row_name)
        max_len_by_column = []
        for column_name in column_names:
            max_len_by_column.append(len(str(column_name)))

        for j in range(len(column_names)):
            for i in range(len(row_names)):
                if max_len_by_column[j] < len(str(list_of_rows[i][j])):
                    max_len_by_column[j] = len(str(list_of_rows[i][j]))

        if self.fancy_printing:
            fancy_header_str = st(table_name, max_len_row_names + header_separation) + fancy_vert_sep
            for i in range(len(column_names)):
                fancy_header_str += st(column_names[i], max_len_by_column[i] + header_separation)
                if i in subtable_borders:
                    fancy_header_str += fancy_vert_sep
        plain_header_str = st(table_name, max_len_row_names + header_separation) + plain_vert_sep
        header_length = max_len_row_names + header_separation + vert_sep_len
        for i in range(len(column_names)):
            plain_header_str += st(column_names[i], max_len_by_column[i] + header_separation)
            header_length += max_len_by_column[i] + header_separation
            if i in subtable_borders:
                plain_header_str += plain_vert_sep
                header_length += vert_sep_len

        if self.fancy_printing:
            fancy_print_queue.append(fancy_header_str)
        plain_print_queue.append(plain_header_str)
        if self.fancy_printing:
            fancy_print_queue.append(color.BRIGHT_YELLOW + "-" * header_length + color.END)
        plain_print_queue.append("-" * header_length)
        for i in range(len(row_names)):
            if self.fancy_printing:
                fancy_cur_str = st(row_names[i], max_len_row_names + header_separation) + fancy_vert_sep
                for j in range(len(column_names)):
                    fancy_cur_str += st(list_of_rows[i][j], max_len_by_column[j] + header_separation)
                    if j in subtable_borders:
                        fancy_cur_str += fancy_vert_sep
                fancy_print_queue.append(fancy_cur_str)
            plain_cur_str = st(row_names[i], max_len_row_names + header_separation) + plain_vert_sep
            for j in range(len(column_names)):
                plain_cur_str += st(list_of_rows[i][j], max_len_by_column[j] + header_separation)
                if j in subtable_borders:
                    plain_cur_str += plain_vert_sep
            plain_print_queue.append(plain_cur_str)
        if self.fancy_printing:
            fancy_print_queue.append(color.BRIGHT_YELLOW + "-" * header_length + color.END)
        plain_print_queue.append("-" * header_length)

        for i in range(len(plain_print_queue)):
            if self.fancy_printing:
                self.write(fancy_print_queue[i], plain_msg = plain_print_queue[i])
            else:
                self.write(plain_print_queue[i])



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

                if self.fancy_printing:
                    tau_space = kwargs["tau_space"]
                else:
                    # We transform tau_space into a special array
                    smallest_tau = kwargs["tau_space"][1]
                    largest_tau = kwargs["tau_space"][-1]

                    tau_coef = pow(largest_tau / smallest_tau, 1 / (self.plain_semaphor_N + 1))

                    tau_space = [0, smallest_tau]

                    for i in range(self.plain_semaphor_N):
                        tau_space.append(tau_space[-1] * tau_coef)
                    tau_space.append(largest_tau)

                semaphor_event_ID, semaphor_header = self.semaphor.create_event(tau_space, header, s_nl)
                self.cur_semaphor_event = semaphor_event_ID
                self.write(semaphor_header, v)

        else:
            self.write(header, v)
        self.depth += 1

        if semaphored and self.fancy_printing:
            # We create the placeholder line
            self.write(header + " waiting for the first event...", v)

    def update_semaphor_event(self, tau):
        if self.print_on_the_fly and self.cur_semaphor_event is not None:
            msg = self.semaphor.update(self.cur_semaphor_event, tau)
            if msg is not None:
                # We don't write update messages into the log which meant to be saved, only to print on the fly
                if self.fancy_printing:
                    print(self.pipe_stack(self.depth - 1) + self.spinning_pipe() + self.cur_highlight(msg), end='\r')
                    self.last_printed_line = msg
                    self.cur_semaphor_prefix_id = (self.cur_semaphor_prefix_id + 1) % len(Journal.semaphor_prefixes)
                else:
                    print(self.pipe_stack(self.depth) + msg)
            elif self.fancy_printing:
                # We just reprint the last statement with a rotated pipe
                print(self.pipe_stack(self.depth - 1) + self.spinning_pipe() + self.cur_highlight(self.last_printed_line), end='\r')

    def exit(self, end_message = None):
        # If popping a semaphor, it will return the duration in ms
        last_routine = self.routine_stack.pop()


        if end_message is not None:
            exit_msg = " " + end_message
        else:
            exit_msg = ""

        # check if it's semaphored
        was_it_semaphored = self.cur_semaphor_event is not None
        if was_it_semaphored:
            # We exit the semaphored routine
            process_duration, semaphor_exit_msg = self.semaphor.finish_event(self.cur_semaphor_event, end_message)
            self.cur_semaphor_event = None
            self.cur_semaphor_prefix_id = 0
            exit_msg = " " + semaphor_exit_msg
        self.depth -= 1
        if self.fancy_printing and self.verbosity >= last_routine[1]:
            self.write(f"\\_{exit_msg}", last_routine[1], prevent_printing_on_the_fly = True)
            if self.print_on_the_fly:
                if not was_it_semaphored:
                    self.commit_last_line() # we don't want to hang the last semaphor update
                self.preview_print(gradual_fg_color[self.depth % len(gradual_fg_color)] + "\\_" + color.END + exit_msg)
        else:
            self.write(f"\\_{exit_msg}", last_routine[1])
        if was_it_semaphored:
            return(process_duration)

    # Interactive element

    def ask_yes_no(self, question):
        # Requires a print on the fly method.
        # Doesn't proceed until given answer
        # Returns True if yes, False if no

        if self.yes is not None:
            if self.yes:
                self.write(question + " (Automatically decided 'yes')")
            else:
                self.write(question + " (Automatically decided 'no')")
            return(self.yes)

        if not self.print_on_the_fly:
            # Assume yes
            self.write(question + " (Assumed 'yes')")
            return(True)

        # First, we commit the previous line
        self.commit_last_line()

        ans = "NOT_PROVIDED"

        question_print = self.pre() + question + " (y / n) "
        if self.fancy_printing:
            question_print = self.pre() + self.cur_highlight(question) + " (" + color.GREEN + "y" + color.END + " / " + color.RED + "n" + color.END + ") "

        while ans.lower() not in Journal.valid_answers:
            ans = input(question_print)

        if ans in Journal.answer_strings["yes"]:
            self.write(question + "(Answered 'yes')", prevent_printing_on_the_fly = True)
            return(True)
        if ans in Journal.answer_strings["no"]:
            self.write(question + "(Answered 'no')", prevent_printing_on_the_fly = True)
            return(False)

# -----------------------------------------------------------------------------
# --------------------------- class DisabledJournal ---------------------------
# -----------------------------------------------------------------------------
# Pass this as a Journal to prevent printing without throwing exceptions on
# method calls

class DisabledJournal():
    def __init__(self, verbosity = 0, print_on_the_fly = True, fancy_printing = True, max_row_width = 200):
        pass

    def write(self, msg, v = -1, prevent_printing_on_the_fly = False):
        pass

    def enter(self, header, v = -1, semaphored = False, **kwargs):
        pass

    def exit(self, end_message = None):
        pass

    def print_matrix(self, m, label, dec_points = 5, max_rows = 11):
        pass

    def print_table(self, table_name, column_names, row_names, list_of_rows, subtable_borders = [], header_separation = 2):
        pass

    def update_semaphor_event(self, tau):
        pass

