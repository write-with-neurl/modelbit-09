import modelbit, sys
from typing import *

# main function
def zephyr_prompt(prompt):
    return {'output': 'Hello World!'}

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(zephyr_prompt(*(modelbit.parseArg(v) for v in sys.argv[1:])))