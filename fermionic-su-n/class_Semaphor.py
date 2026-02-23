# Like Semaphor but cool
# Can create multiple events and thread them
#
# How does this work?
# Whenever you want to semaphorize a function, make sure it takes "semaphor_event_id" as an argument,
# and add [semaphor instance].update(semaphor_event_id) at the very end of the function.
# Before calling the function, create a new event with a custom message, which yields a unique event ID,
# and then pass it into the semaphorised function as an argument.
# After the function wrapper concludes, finish the event.
#
# Example pseudocode:
#
# my_semaphor = new Semaphor()
#
# def func(..., semaphor_event_id):
#     ...
#     my_semaphor.update(semaphor_event_id)
#     return(...)
#
# new_event_ID = my_semaphor.create_event(tau_space)
# while(...):
#     ... = func(..., new_event_ID)
#     ...
#
#

import time

def dtstr(seconds, max_depth = 2):
    # Dynamically chooses the right format
    # max_depth is the number of different measurements (e.g. max_depth = 2: "2 days 5 hours")
    if seconds >= 60 * 60 * 24:
        # Days
        if max_depth == 1:
            return(f"{int(round(seconds / (60 * 60 * 24)))} days")
        remainder = seconds % (60 * 60 * 24)
        days = int((seconds - remainder) / (60 * 60 * 24))
        return(f"{days} days {dtstr(remainder, max_depth - 1)}")
    if seconds >= 60 * 60:
        # Hours
        if max_depth == 1:
            return(f"{int(round(seconds / (60 * 60)))} hours")
        remainder = seconds % (60 * 60)
        hours = int((seconds - remainder) / (60 * 60))
        return(f"{hours} hours {dtstr(remainder, max_depth - 1)}")
    if seconds >= 60:
        # Minutes
        if max_depth == 1:
            return(f"{int(round(seconds / 60))} min")
        remainder = seconds % (60)
        minutes = int((seconds - remainder) / (60))
        return(f"{minutes} min {dtstr(remainder, max_depth - 1)}")
    if seconds >= 1:
        # Seconds
        if max_depth == 1:
            return(f"{int(round(seconds))} sec")
        remainder = seconds % (1)
        secs = int((seconds - remainder))
        return(f"{secs} sec {dtstr(remainder, max_depth - 1)}")
    # Milliseconds
    return(f"{int(round(seconds / 0.001))} ms")

class Semaphor():

    def __init__(self, time_format = "%H:%M:%S", print_directly = True):
        self.time_format = time_format
        self.print_directly = print_directly
        self.N_events = 0

        # "Tau" is simulation time, "t" is real time
        self.tau_space = {}
        self.start_tau = {}
        self.start_time = {}
        self.next_flag_tau_index = {}
        self.ETA = {}

        self.message = {}
        self.max_msg_len = {}
        self.newline = {}

    def create_event(self, tau_space, message, newline = False):
        # newline should be set to True for nested events, False for the innermost events or non-nested events

        # New ID
        new_ID = f"S_EVENT_{self.N_events}"
        self.N_events += 1

        self.tau_space[new_ID] = tau_space
        self.start_tau[new_ID] = tau_space[0]
        self.start_time[new_ID] = time.time()
        self.next_flag_tau_index[new_ID] = 1
        self.ETA[new_ID] = None

        self.message[new_ID] = message
        self.max_msg_len[new_ID] = 0
        self.newline[new_ID] = newline

        if self.print_directly:
            print(f"{message} begins at {time.strftime(self.time_format, time.localtime( self.start_time[new_ID]))}")
        return(new_ID, f"{message} begins at {time.strftime(self.time_format, time.localtime( self.start_time[new_ID]))}")

    def finish_event(self, event_ID, final_message = None):
        # Returns the duration in ms
        if event_ID not in self.tau_space.keys():
            #print(f"  ERROR: Semaphor {event_ID} does not exist.")
            return(-1)
        if final_message is None:
            final_message = self.message[event_ID]

        process_duration = time.time() - self.start_time[event_ID]

        msg = f"{final_message} finished at {time.strftime(self.time_format, time.localtime( time.time()))} (duration {dtstr(process_duration)})"
        if len(msg) < self.max_msg_len[event_ID]:
            # padding
            msg += " " * (self.max_msg_len[event_ID] - len(msg))
        del self.tau_space[event_ID]
        del self.start_tau[event_ID]
        del self.start_time[event_ID]
        del self.next_flag_tau_index[event_ID]
        del self.ETA[event_ID]

        del self.message[event_ID]
        del self.max_msg_len[event_ID]
        del self.newline[event_ID]
        if self.print_directly:
            print(msg)
            return(process_duration)
        else:
            return(process_duration, msg)

    def update(self, event_ID, tau):
        if event_ID not in self.tau_space.keys():
            #print(f"  ERROR: Semaphor {event_ID} does not exist.")
            if self.print_directly:
                print(f"  ERROR: Semaphor {event_ID} does not exist.")
                return(-1)
            else:
                return(f"  ERROR: Semaphor {event_ID} does not exist.")
        # check if semaphor finished
        if self.next_flag_tau_index[event_ID] >= len(self.tau_space[event_ID]):
            if self.newline[event_ID]:
                if self.print_directly:
                    print("  Semaphor reached the final flagged timestamp. No further semaphor update necessary.")
                    return(0)
                else:
                    return("  Semaphor reached the final flagged timestamp. No further semaphor update necessary.")
            else:
                msg = "  Semaphor reached the final flagged timestamp. No further semaphor update necessary."
                if len(msg) < self.max_msg_len[event_ID]:
                    # padding
                    msg += " " * (self.max_msg_len[event_ID] - len(msg))
                if self.print_directly:
                    print(msg, end='\r')
                    return(0)
                else:
                    return(msg)
        if tau >= self.tau_space[event_ID][self.next_flag_tau_index[event_ID]]:
            # We find the next smallest unreached semaphor flag
            tau_index_new = self.next_flag_tau_index[event_ID]
            while(self.tau_space[event_ID][tau_index_new] <= tau):
                tau_index_new += 1
                if tau_index_new >= len(self.tau_space[event_ID]):
                    break
            self.next_flag_tau_index[event_ID] = tau_index_new
            progress_fraction = (self.tau_space[event_ID][tau_index_new - 1] - self.start_tau[event_ID]) / (self.tau_space[event_ID][-1] - self.start_tau[event_ID])
            ETA = time.strftime(self.time_format, time.localtime( (time.time()-self.start_time[event_ID]) / progress_fraction + self.start_time[event_ID] ))

            if self.newline[event_ID]:
                if self.print_directly:
                    print(f"  {self.message[event_ID]}: {str(int(100 * progress_fraction)).zfill(2)}% done; est. time of finish: {ETA} (sim. t = {tau:.2f})")
                    return(0)
                else:
                    return(f"  {self.message[event_ID]}: {str(int(100 * progress_fraction)).zfill(2)}% done; est. time of finish: {ETA} (sim. t = {tau:.2f})")
            else:
                msg = f"  {self.message[event_ID]}: {str(int(100 * progress_fraction)).zfill(2)}% done; est. time of finish: {ETA} (sim. t = {tau:.2f})"
                if len(msg) < self.max_msg_len[event_ID]:
                    # padding
                    msg += " " * (self.max_msg_len[event_ID] - len(msg))
                self.max_msg_len[event_ID] = len(msg)

                if self.print_directly:
                    print(msg, end='\r')
                    return(0)
                else:
                    return(msg)
        return(None)



