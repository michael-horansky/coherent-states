# This is a document with all the typographical and numerical constants which need to be consistent across all the game files

# -----------------------------------------------------------------------------
# -------------------------- Typographical constants --------------------------
# -----------------------------------------------------------------------------

# -------------------------------- Delimiters ---------------------------------
# Since representations have to be nested, these delimiters have to be unique!
STPos_delim = ","
Flag_delim = ":"
Gamemaster_delim = ";"



# ---------------------------------- Colors -----------------------------------
# ANSI escape codes (SGR parameters)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
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

# ----------------------------- Legacy constants ------------------------------
stone_type_representations = {
        "T" : ["A", "tank"],
        "L" : ["B", "tank"],
        "Y" : ["A", "bombardier"],
        "V" : ["B", "bombardier"],
        "C" : ["A", "tagger"],
        "G" : ["B", "tagger"],
        "S" : ["A", "sniper"],
        "Z" : ["B", "sniper"],
        "W" : ["A", "wildcard"],
        "M" : ["B", "wildcard"],
        "#" : ["GM", "box"],
        "@" : ["GM", "mine"]
    }

stone_symbols = {
        "A" : {
                "tank" : "T",
                "bombardier" : "Y",
                "tagger" : "C",
                "sniper" : "S",
                "wildcard" : "W"
            },
        "B" : {
                "tank" : "L",
                "bombardier" : "V",
                "tagger" : "G",
                "sniper" : "Z",
                "wildcard" : "M"
            },
        "GM" : {
                "box" : "#",
                "mine" : "@"
            }
    }


base_representations = {
        "A" : "A",
        "B" : "B",
        "!" : "neutral"
    }

base_symbols = {
        "A" : "A",
        "B" : "B",
        "neutral" : "!"
    }
