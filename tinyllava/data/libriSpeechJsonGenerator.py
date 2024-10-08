import shortuuid
import random
import json
import os
from tqdm import tqdm
import re
import soundfile as sf

# generated by gpt4o
description_list = [
    "Transcribe the audio content.",
    "Provide a transcription of the given audio.",
    "Convert the speech from the audio to text.",
    "Accurately transcribe the spoken words in the audio.",
    "Generate a text version of the speech in the audio file.",
    "Produce a clear transcription of the audio content.",
    "Convert the spoken language in the audio to written text.",
    "Document the speech in the audio as a text transcript.",
    "Write down what is spoken in the provided audio.",
    "Translate the audio speech into a text format.",
    "Deliver an accurate transcription of the audio clip.",
    "Capture the spoken words in the audio as text.",
    "Convert the contents of the audio into text form.",
    "Transcribe the verbal content of the audio.",
    "Provide a word-for-word transcription of the audio.",
    "Record the speech in the audio as a text document.",
    "Produce a textual representation of the spoken words in the audio.",
    "Transcribe the speech accurately from the audio file.",
    "Generate a text transcription of the provided audio.",
    "Translate the verbal communication in the audio into text.",
    "Create a precise transcript of the speech from the audio.",
    "Capture the essence of the spoken words in the audio as text.",
    "Transform the audio's spoken content into a written transcript.",
    "Offer a text version of the speech captured in the audio.",
    "Record the audio's spoken words in a written format.",
    "Render the speech from the audio file into text.",
    "Document the audio speech as a text transcription.",
    "Provide a detailed transcription of the audio speech.",
    "Translate the audio recording's speech into text.",
    "Transcribe the provided audio into text format.",
    "Turn the audio into a text transcript.",
    "Accurately convert the spoken audio into text.",
    "Document the contents of the audio recording in text.",
    "Write out the words spoken in the audio file.",
    "Generate a precise transcription of the audio.",
    "Provide a clear and accurate transcription of the speech.",
    "Convert the spoken language from the audio into text.",
    "Transcribe the contents of the audio file to text.",
    "Create a text version of the speech in the audio recording.",
    "Capture the audio's speech as text.",
    "Convert the spoken words in the audio file to written text.",
    "Transcribe the provided audio recording.",
    "Translate the speech from audio into text form.",
    "Render the spoken content of the audio in text.",
    "Document the audio's verbal communication in text.",
    "Produce an accurate transcription of the given audio clip.",
    "Convert the audio speech into a text document.",
    "Write a text representation of the speech from the audio.",
    "Transcribe the words spoken in the audio accurately.",
    "Create a word-for-word transcription of the audio content.",
    "Capture the speech in the audio as a text transcript.",
    "Generate text from the spoken content in the audio.",
    "Transcribe the audio recording into a text format.",
    "Provide an exact transcription of the speech in the audio.",
    "Convert the spoken content in the audio into a text transcript.",
    "Document the verbal content of the audio as text.",
    "Create a precise text record of the audio's speech.",
    "Translate the spoken audio into a written transcript.",
    "Transcribe the audio speech clearly and accurately.",
    "Write out a transcription of the words in the audio recording."
]

dev_clean_path = "/home/user27/LibriSpeech/dev-clean"
dev_other_path = "/home/user27/LibriSpeech/dev-other"
test_clean_path = "/home/user27/LibriSpeech/test-clean"
test_other_path = "/home/user27/LibriSpeech/test-other"
train100_path = "/home/user27/LibriSpeech/train-clean-100"
train360_path = "/home/user27/LibriSpeech/train-clean-360"
train500_path = "/home/user27/LibriSpeech/train-other-500"


def obtain_transcribation(file, data_dict, current_path):
    with open(file, 'r') as f:
        for line in f.readlines():
            sample, transcribation = line.split(" ", 1)
            sample_path = f"{current_path}/{sample}.flac"
            data_dict[sample_path] = transcribation.strip().lower()


def generate_multimodal_ds(split_dict, split_type, get_lengths):
    audio_data = []
    for sample in tqdm(split_dict.keys(), desc=split_type):
        id_data = re.findall("[0-9]+-[0-9]+-[0-9]+", sample)
        sample_dict = dict()
        sample_dict["id"] = id_data
        sample_dict["audio"] = sample
        conversations = [
            {"from": "human", "value": "<audio>\n" + random.choice(description_list)},
            {"from": "gpt", "value": split_dict[sample]}
        ]
        sample_dict['conversations'] = conversations
        if get_lengths:
            wavs, sampling_rate = sf.read(sample)
            sample_dict["length"] = len(wavs)
        audio_data.append(sample_dict)
    return audio_data


def generate_dict(path, split_type, get_lengths=True):
    all_paths = {}
    for speaker_id in os.listdir(path):
        current_path = path+"/"+speaker_id
        for chapter in os.listdir(current_path):
            transcribe_path = f"{path}/{speaker_id}/{chapter}/{speaker_id}-{chapter}.trans.txt"
            obtain_transcribation(transcribe_path, all_paths, f"{current_path}/{chapter}")
    return generate_multimodal_ds(all_paths, split_type, get_lengths)
        

dev_clean_data = generate_dict(dev_clean_path, "dev clean")
with open(f"{dev_clean_path}.json", 'w') as f: json.dump(dev_clean_data, f, indent=4)


dev_other_data = generate_dict(dev_other_path, "dev other")
with open(f"{dev_other_path}.json", 'w') as f: json.dump(dev_other_data, f, indent=4)


test_clean_data = generate_dict(test_clean_path, "test clean")
with open(f"{test_clean_path}.json", 'w') as f: json.dump(test_clean_data, f, indent=4)


test_other_data = generate_dict(test_other_path, "test other")
with open(f"{test_other_path}.json", 'w') as f: json.dump(test_other_data, f, indent=4)


train100_data = generate_dict(train100_path, "train100")
train360_data = generate_dict(train360_path, "train360")
train500_data = generate_dict(train500_path, "train500")
train_data = train100_data + train360_data + train500_data
with open(f"/home/user27/LibriSpeech/train.json", 'w') as f: json.dump(train_data, f, indent=4)


assert len(dev_clean_data) == 2703
assert len(dev_other_data) == 2864
assert len(test_clean_data) == 2620
assert len(test_other_data) == 2939
assert len(train100_data) == 28539
assert len(train360_data) == 104014
assert len(train500_data) == 148688
print("All tests run succesfully")







