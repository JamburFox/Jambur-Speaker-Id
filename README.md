# Jambur-Speaker-Id
A Pytorch project which predicts speaker id from embeddings

# How to run model
run `python run.py --audio_file ./test.wav`
## Variables:
- `--audio_file` the location of the file to check
- `--device` the device to use (this variable is automatically set but can be overridden)

# How to create voice embedding
run `python create_voice_embedding.py --speaker_id user --audio_file ./test.wav`
## Variables:
- `--speaker_id` the id of the speaker to save the embedding for
- `--audio_file` the location of the file
- `--device` the device to use (this variable is automatically set but can be overridden)