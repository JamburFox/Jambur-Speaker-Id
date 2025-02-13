# Jambur-Speaker-Id
A Pytorch project which predicts speaker id from embeddings

# How to run model
run `python run.py --audio_file ./test.wav`
### Variables:
- `--audio_file` the location of the file to check
- `--device` the device to use (this variable is automatically set but can be overridden)

# How to create voice embedding
run `python create_voice_embedding.py --speaker_id user --audio_file ./test.wav`
### Variables:
- `--speaker_id` the id of the speaker to save the embedding for
- `--audio_file` the location of the file to add
- `--device` the device to use (this variable is automatically set but can be overridden)


# How to train model
run `python train.py`
### Variables:
modify the variables directly inside the python script: EPOCHS, BATCH_SIZE and LEARNING_RATE

### Dataset
the format of the dataset should be:
- /dataset/speakers.csv
- /dataset/audio/...
The speakers.csv file should contain a new line for each unique audio file. Each line should for formatted like: "file,label" e.x: "audiofile.wav,0"
and the corrosponding audio file should be in the audio folder