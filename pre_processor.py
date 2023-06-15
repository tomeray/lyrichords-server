import librosa
from transformers import Wav2Vec2FeatureExtractor
import numpy as np


def split_audio_into_segments(received_audio, segment_length=10):
    # Load the audio file with librosa
    audio_file, sr = librosa.load(received_audio, sr=24000)
    segments = []

    # Calculate the number of samples in each segment
    samples_per_segment = int(segment_length * sr)

    #  Calculate the number of segments
    num_segments = int((len(audio_file) - samples_per_segment) / samples_per_segment) + 1

    # Split the audio into segments with overlap
    for i in range(num_segments):
        segment_start = i * samples_per_segment
        segment_end = segment_start + samples_per_segment
        segment = audio_file[segment_start:segment_end]

        # Save the segment as an MP3 file in the new_audio directory
        segments.append(segment)

    reminder = audio_file[num_segments * samples_per_segment + 1:]
    segments.append(np.concatenate((reminder, np.zeros(samples_per_segment - len(reminder))), axis=0))

    return segments


class PreProcessor:

    def __init__(self):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M")

    def preprocess_segment(self, audio_array):
        # Load audio file and convert it to tensor
        input_features = self.processor(audio_array,
                                        sampling_rate=self.processor.sampling_rate,
                                        return_tensors="pt"
                                        )
        # Load lab file and encode it to codebook index
        input_features['input_values'] = input_features.input_values
        input_features['attention_mask'] = input_features.attention_mask
        # Return input and label pair
        return input_features

    def preprocess(self, audio_file):
        segments = split_audio_into_segments(audio_file)
        preprocessed_segments = []
        for segment in segments:
            preprocessed_segments.append(self.preprocess_segment(segment))

        return preprocessed_segments
