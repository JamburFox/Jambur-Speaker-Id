# Jambur-Speaker-Id
A Pytorch project which predicts speaker id from embeddings


# Tutorial (Command Line)
This is a tutorial to show how to setup and run this project.
note: The command's shown should be run in the root directory of this project.

First we need to create some voice embeddings so that the program knows who each voice belongs to.
We can do this by running the command `python create_voice_embedding.py --speaker_id user --audio_file ./test.wav` where "user" can be whatever string you want for the speaker id and "./test.wav" can be replaced with wherever the audio file is located (this can also be a url to a folder to scan all files for that speaker)

After that we can run the command `python run.py --audio_file ./test.wav` where "./test.wav" can be replaced with the location of the audio file you want to check, this will output the closest matching speaker id to that voice.


# Tutorial (Python)
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


# How to create voice embedding
run `python create_voice_embedding.py --speaker_id user --audio_file ./test.wav`

Instead of an audio file you can also specify a folder and it will create an embedding for all audio files in that folder
### Arguments:
- `--speaker_id` the id of the speaker to save the embedding for
- `--audio_file` the location of the file or folder to scan
- `--model` location of the model (this variable is automatically set to the default model)
- `--device` the device to use (this variable is automatically set but can be overridden)


# How to run model
run `python run.py --audio_file ./test.wav`
### Arguments:
- `--audio_file` the location of the file to check
- `--model` location of the model (this variable is automatically set to the default model)
- `--device` the device to use (this variable is automatically set but can be overridden)


# How to train model
run `python train.py --dataset ./dataset --epochs 10 --batch_size 32 --learning_rate 1e-3`
### Arguments:
- `--dataset` location of the dataset
- `--epochs` number of epochs to train for
- `--batch_size` the batch size
- `--learning_rate` the learning rate
- `--scheduler_step` the number of epochs before multiplying learning rate by gamma
- `--scheduler_gamma` the amount to multiply the learning rate by upon x scheduler steps (e.x. 0.1 * 0.5 = 0.05)
- `--device` the device to use (this variable is automatically set but can be overridden)
### Dataset
the format of the dataset should be:
- ./dataset/speakers.csv
- ./dataset/audio/...

The speakers.csv file should contain a new line for each unique audio file. Each line should for formatted like: "file,label" e.x: "audiofile.wav,0"
and the corrosponding audio file should be in the audio folder


# TODO
- Save multiple embeddings under a single .npy file to reduce loading time.
- When running the project, compare multiple embeddings under a single operation similar to whats done in train.py (compare_embeddings_cosine()) to increase efficiency and get the highest similarities from the result along with its corrsponding indexed speaker as the outcome.
- When creating new embeddings with multiple files from a folder, batch these files together and infer them all at once on the model. This will have a batch limit variable so that if there are more files than the limit it will run under multiple batches e.x. if the limit is 16, 30 files would be batched as 16 and 14.
- Model parameters will automatically adjust to specified model using the .json metadata file
- When running the project add a threshold variable so that if no embedding comparison has reached this threshold then return an unknown speaker.
- Add dynamic embedding saving: "run_model_dynamic_fill(audio_file: str, speaker: str, total_embeddings: int)" This will run the model as normal returning the speaker id but will also automatically expand the embedding list of that speaker if the amount of embeddings that speaker has is below a certain amount. If this embedding is an unknown speaker it will automatically create a new speaker and save the embedding.
If the total embeddings after saving is above the total_embeddings variable it will automatically delete the embedding which has the lowest similarity score to the saved one.