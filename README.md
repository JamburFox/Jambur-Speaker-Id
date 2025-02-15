# Jambur-Speaker-Id
A Pytorch project which predicts speaker id from embeddings


# Tutorial
This is a tutorial to show how to setup and run this project.
note: The command's shown should be run in the root directory of this project.

First we need to create some voice embeddings so that the program knows who each voice belongs to.
We can do this by running the command `python create_voice_embedding.py --speaker_id user --audio_file ./test.wav` where "user" can be whatever string you want for the speaker id and "./test.wav" can be replaced with wherever the audio file is located (this can also be a url to a folder to scan all files for that speaker)

After that we can run the command `python run.py --audio_file ./test.wav` where "./test.wav" can be replaced with the location of the audio file you want to check, this will output the closest matching speaker id to that voice.

# Python
To use this package within another python project copy the "jambur_speaker_id" folder into the directory of your existing project. Then you can simply import "load_embedding_model" and "run_model_file" or "run_model_audio" from "jambur_speaker_id.model_manager" and use these functions to run the model.

```python
from jambur_speaker_id.model_manager import load_embedding_model, run_model_file, run_model_audio #import files

device = "cpu" # can replace cpu with cuda to run on gpu or any other pytorch supported device
model = load_embedding_model().to(device) # load model
speaker_id = run_model_file(model, "./test.wav", device, False) # run model by loading an audio file

import librosa
audio, sr = librosa.load("./test.wav", sr=None) # load audio
speaker_id = run_model_audio(model, audio, sr, device, False) # run model by scanning loaded audio directly
```


# How to run model
run `python run.py --audio_file ./test.wav`
### Arguments:
- `--audio_file` the location of the file to check
- `--device` the device to use (this variable is automatically set but can be overridden)

# How to create voice embedding
run `python create_voice_embedding.py --speaker_id user --audio_file ./test.wav`

Instead of an audio file you can also specify a folder and it will create an embedding for all audio files in that folder
### Arguments:
- `--speaker_id` the id of the speaker to save the embedding for
- `--audio_file` the location of the file or folder to scan
- `--device` the device to use (this variable is automatically set but can be overridden)


# How to train model
run `python train.py --epochs 10 --batch_size 32 --learning_rate 1e-3`
### Arguments:
- `--epochs` number of epochs to train for
- `--batch_size` the batch size
- `--learning_rate` the learning rate
- `--device` the device to use (this variable is automatically set but can be overridden)

### Dataset
the format of the dataset should be:
- ./dataset/speakers.csv
- ./dataset/audio/...

The speakers.csv file should contain a new line for each unique audio file. Each line should for formatted like: "file,label" e.x: "audiofile.wav,0"
and the corrosponding audio file should be in the audio folder