import modelbit, sys
from typing import *
from tempfile import NamedTemporaryFile
from whisper.model import Whisper
import requests

model = modelbit.load_value("data/model.pkl") # Whisper( (encoder): AudioEncoder( (conv1): Conv1d(80, 768, kernel_size=(3,), stride=(1,), padding=(1,)) (conv2): Conv1d(768, 768, kernel_size=(3,), stride=(2,), padding=(1,)) (blocks): ModuleList( (0-...

# main function
def whisper_transcribe(file):
    # Download the file data
    response = requests.get(file)

    # Just ensure that the download was successful
    response.raise_for_status()

    with NamedTemporaryFile() as temp:
        # Write the downloaded data to the temporary file
        temp.write(response.content)
        temp.flush()

        # Let's get the transcript of the temporary file.
        transcript = model.transcribe(temp.name)

        return { 'transcript': transcript['text'] }

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(whisper_transcribe(*(modelbit.parseArg(v) for v in sys.argv[1:])))