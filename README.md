# Jambur-Speaker-Id
A Pytorch project which predicts speaker id from embeddings

# How to run model
run `python run.py --audio_file ./test.wav`
### Variables:
- `--audio_file` the location of the file to check
- `--device` the device to use (this variable is automatically set but can be overridden)

# How to create voice embedding
run `python create_voice_embedding.py --speaker_id user --audio_file ./test.wav`
Instead of an audio file you can also specify a folder and it will create an embedding for all audio files in that folder
### Variables:
- `--speaker_id` the id of the speaker to save the embedding for
- `--audio_file` the location of the file to add
- `--device` the device to use (this variable is automatically set but can be overridden)


# How to train model
run `python train.py --epochs 10 --batch_size 32 --learning_rate 1e-3`
### Variables:
- `--epochs` number of epochs to train for
- `--batch_size` the batch size
- `--learning_rate` the learning rate
- `--device` the device to use (this variable is automatically set but can be overridden)

### Dataset
the format of the dataset should be:
- /dataset/speakers.csv
- /dataset/audio/...

The speakers.csv file should contain a new line for each unique audio file. Each line should for formatted like: "file,label" e.x: "audiofile.wav,0"
and the corrosponding audio file should be in the audio folder