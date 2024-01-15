import modelbit, sys
from typing import *
from functools import cache
import pickle

@cache
def get_llm():
    with open('falcon_pipe_int4.pkl', 'rb') as file:
        content = pickle.load(file)
    return content


# main function
def run_falcon_prompt(prompt):
    falcon_pipe = get_llm()
    sequences = falcon_pipe(
        prompt,
        do_sample=False,
        batch_size=8,
        max_new_tokens=50,
        temperature=0.7,
        top_k=10,
        num_return_sequences=1,
    )
    return {'output': sequences[0]['generated_text']}

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(run_falcon_prompt(*(modelbit.parseArg(v) for v in sys.argv[1:])))