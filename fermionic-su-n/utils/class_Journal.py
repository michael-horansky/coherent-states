
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

    def __init__(self, verbosity = 0, print_on_the_fly = True, fancy_printing = True, max_row_width = 200):

        self.journal_text = ""
        self.routine_stack = [] # every routine is [header, max required verbosity]
        self.verbosity = verbosity # The higher the verbosity, the more in detail the Journal is
        self.print_on_the_fly = print_on_the_fly # If True, printing to terminal happens concurrently
        self.depth = 0

        self.fancy_printing = fancy_printing # if True, we use colors, highlighting etc

        self.max_row_width = max_row_width # used for clever wrapping

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

    def write(self, msg, v = -1, prevent_printing_on_the_fly = False):
        # Writes a message inside current routine at current depth
        if self.verbosity >= max(v, self.get_stack_v()):
            self.journal_text += self.pre(False) + msg + "\n"
            if self.print_on_the_fly and not prevent_printing_on_the_fly:
                if self.fancy_printing:
                    # We firstly print the last submitted line, and then the preprint of the current line
                    self.commit_last_line()
                    self.preview_print(msg)
                else:
                    # crude printing only
                    print(self.pre() + msg)

    # Special object printing methods

    def print_matrix(self, m, label, dec_points = 5, max_rows = 11):

        # First, we create the label prefix
        print_queue = []
        no_of_rows = min(len(m), max_rows)

        pre_col = []
        if no_of_rows == 1:
            if self.fancy_printing:
                pre_col = [color.BRIGHT_YELLOW + "(" + color.END]
            else:
                pre_col = ["("]
        else:
            if self.fancy_printing:
                pre_col = [color.BRIGHT_YELLOW + "/" + color.END]
                for i in range(no_of_rows - 2):
                    pre_col.append(color.BRIGHT_YELLOW + "|" + color.END)
                pre_col.append(color.BRIGHT_YELLOW + "\\" + color.END)
            else:
                pre_col = ["/"]
                for i in range(no_of_rows - 2):
                    pre_col.append("|")
                pre_col.append("\\")

        for i in range(no_of_rows):
            if i == no_of_rows // 2:
                print_queue.append(f"{label} = {pre_col[i]}")
            else:
                print_queue.append(" " * (len(label) + 3) + f"{pre_col[i]}")


        # Now, we calculate how many numbers we can afford to print to respect the line length
        col_width = [0] * len(m[0])
        for i in range(len(m)):
            for j in range(len(m[0])):
                cur_len = len(str(round(m[i][j], dec_points))) + 2 # the 2 is for minimal padding
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
                        print_queue[i] += st(str(round(m[i][j], dec_points)), col_width[j])
                    print_queue[i] += " " + st(str(round(m[i][len(m[i]) - 1], dec_points)), col_width[-1] - 2)
            else:
                # we trim after max_rows - 2
                for i in range(max_rows - 2):
                    for j in range(len(m[i]) - 1):
                        print_queue[i] += st(str(round(m[i][j], dec_points)), col_width[j])
                    print_queue[i] += " " + st(str(round(m[i][len(m[i]) - 1], dec_points)), col_width[-1] - 2)
                for j in range(len(m[0]) - 1):
                    print_queue[max_rows - 2] += st("...", col_width[j])
                print_queue[max_rows - 2] += " " + st("...", col_width[-1] - 2)
                for j in range(len(m[-1]) - 1):
                    print_queue[max_rows - 1] += st(str(round(m[-1][j], dec_points)), col_width[j])
                print_queue[max_rows - 1] += " " + st(str(round(m[-1][len(m[-1]) - 1], dec_points)), col_width[-1] - 2)
        else:
            # we have to trim the number of entries per row
            if len(m) <= max_rows:
                # we can print all rows
                for i in range(len(m)):
                    for j in range(trim_N):
                        print_queue[i] += st(str(round(m[i][j], dec_points)), col_width[j])
                    print_queue[i] += "... " + st(str(round(m[i][len(m[i]) - 1], dec_points)), col_width[-1] - 2)
            else:
                # we trim rows and columns
                for i in range(max_rows - 2):
                    for j in range(trim_N):
                        print_queue[i] += st(str(round(m[i][j], dec_points)), col_width[j])
                    print_queue[i] += "... " + st(str(round(m[i][len(m[i]) - 1], dec_points)), col_width[-1] - 2)
                for j in range(trim_N):
                    print_queue[max_rows - 2] += st("...", col_width[j])
                print_queue[max_rows - 2] += "    " + st("...", col_width[-1] - 2)
                for j in range(trim_N):
                    print_queue[max_rows - 1] += st(str(round(m[-1][j], dec_points)), col_width[j])
                print_queue[max_rows - 1] += "... " + st(str(round(m[-1][len(m[-1]) - 1], dec_points)), col_width[-1] - 2)

        for line in print_queue:
            self.write(line)




    def print_table(self, table_name, column_names, row_names, list_of_rows, subtable_borders = [], header_separation = 2):
        # column_names[N], row_names[M], list_of_rows[M][N]

        print_queue = [] # each line is an element

        if self.fancy_printing:
            vert_sep = color.BRIGHT_YELLOW + "|" + color.END
        else:
            vert_sep = "|"

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

        header_str = st(table_name, max_len_row_names + header_separation) + vert_sep
        for i in range(len(column_names)):
            header_str += st(column_names[i], max_len_by_column[i] + header_separation)
            if i in subtable_borders:
                header_str += vert_sep

        print_queue.append(header_str)
        if self.fancy_printing:
            print_queue.append(color.BRIGHT_YELLOW + "-" * len(header_str) + color.END)
        else:
            print_queue.append("-" * len(header_str))
        for i in range(len(row_names)):
            cur_str = st(row_names[i], max_len_row_names + header_separation) + vert_sep
            for j in range(len(column_names)):
                cur_str += st(list_of_rows[i][j], max_len_by_column[j] + header_separation)
                if j in subtable_borders:
                    cur_str += vert_sep
            print_queue.append(cur_str)
        if self.fancy_printing:
            print_queue.append(color.BRIGHT_YELLOW + "-" * len(header_str) + color.END)
        else:
            print_queue.append("-" * len(header_str))

        for line in print_queue:
            self.write(line)



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

        if semaphored:
            # We create the placeholder line
            self.write(header + " waiting for the first event...", v)

    def update_semaphor_event(self, tau):
        if self.print_on_the_fly and self.cur_semaphor_event is not None:
            msg = self.semaphor.update(self.cur_semaphor_event, tau)
            if msg is not None:
                # We don't write update messages into the log which meant to be saved, only to print on the fly
                if self.fancy_printing:
                    print(self.pipe_stack(self.depth - 1) + self.spinning_pipe() + self.cur_highlight(msg), end='\r')
                else:
                    print(self.pipe_stack(self.depth - 1) + self.spinning_pipe() + msg, end='\r')
                self.last_printed_line = msg
                self.cur_semaphor_prefix_id = (self.cur_semaphor_prefix_id + 1) % len(Journal.semaphor_prefixes)
            else:
                # We just reprint the last statement with a rotated pipe
                if self.fancy_printing:
                    print(self.pipe_stack(self.depth - 1) + self.spinning_pipe() + self.cur_highlight(self.last_printed_line), end='\r')
                else:
                    print(self.pipe_stack(self.depth - 1) + self.spinning_pipe() + self.last_printed_line, end='\r')
                self.cur_semaphor_prefix_id = (self.cur_semaphor_prefix_id + 1) % len(Journal.semaphor_prefixes)

    def exit(self, end_message = None):
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
            if not was_it_semaphored:
                self.commit_last_line() # we don't want to hang the last semaphor update
            self.preview_print(gradual_fg_color[self.depth % len(gradual_fg_color)] + "\\_" + color.END + exit_msg)
        else:
            self.write(f"\\_{exit_msg}", last_routine[1])

    # Interactive element

    def ask_yes_no(self, question):
        # Requires a print on the fly method.
        # Doesn't proceed until given answer
        # Returns True if yes, False if no

        if not self.print_on_the_fly:
            # Assume yes
            self.write(question + "(Assumed 'yes')")
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

