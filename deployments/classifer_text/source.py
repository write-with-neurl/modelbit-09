import modelbit, sys
from typing import *
from transformers.pipelines import pipeline

# main function
def classifer_text(data):
  classifer_pipeline = pipeline(task='text-classification', model="wesleyacheng/sms-spam-classification-with-bert")
  outputs = classifer_pipeline(data)
 # Simplifying the output
  for item in outputs:
      label = item['label']
      # Round the score to two decimal places and convert to percentage
      score = round(item['score'], 2)
      print(f"{label}: {score}")

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(classifer_text(*(modelbit.parseArg(v) for v in sys.argv[1:])))