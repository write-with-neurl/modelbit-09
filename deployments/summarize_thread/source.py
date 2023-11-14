import modelbit, sys
from typing import *
from functools import cache
from langchain.llms.llamacpp import LlamaCpp
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.stdout import StdOutCallbackHandler

file_path = 'llama-2-7b.Q4_0.gguf'
prompt = modelbit.load_value("data/prompt.pkl") # input_variables=['num_of_words', 'text'] template="\n\n### Instruction:\nGiven a twitter thread: '{text}', You are tasked with summarizing the twitter thread in {num_of_words} words and ensure the sum...

@cache
def load_llm():
    """ Loads the LlamaCpp model with specified model path and context size. 
        Uses caching (`@cache`) to optimize performance by storing the result for subsequent calls.
    """
    llm = LlamaCpp(model_path=file_path, n_ctx=8191) # Load the LlamaCpp model
    return llm


# main function
def summarize_thread(text: str, num_of_words: int = 200):
    """ Summarizes the given text using the LlamaCpp model.

    Args:
        text (str): Text to be summarized.
        num_of_words (int, optional): Number of words for the summary. Defaults to 200.

    Returns:
        The summary of the text.
    """
    llm = load_llm()  # Load the LlamaCpp model
    # Initialize LLMChain with the LlamaCpp model and the provided text as prompt

    chain = LLMChain(llm=llm, prompt=prompt)
    # Run the chain with the text and word limit, and return the result
    return chain.run({"text": text, "num_of_words":num_of_words}, callbacks=[StdOutCallbackHandler()])

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(summarize_thread(*(modelbit.parseArg(v) for v in sys.argv[1:])))