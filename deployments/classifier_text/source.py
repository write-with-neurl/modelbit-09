import modelbit, sys
from typing import *
from transformers.pipelines import pipeline

# main function
def classifier_text(data):
    classifier_pipeline = pipeline(task='text-classification', model="wesleyacheng/sms-spam-classification-with-bert")
    outputs = classifier_pipeline(data)
    for item in outputs:
        label = item['label']
        score = item['score']
        result_dict = {'label': label, 'score': score}  # Create a dictionary for each item
        print(result_dict)  # Print the dictionary in the desired format

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(classifier_text(*(modelbit.parseArg(v) for v in sys.argv[1:])))