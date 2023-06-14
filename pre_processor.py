import librosa
import numpy as np

class PreProcessor:
    def split_audio_into_segments(self, received_audio, segment_length=10):
        # Load the audio file with librosa
        audio_file, sr = librosa.load(received_audio, sr=24000)
        segments = []

        # Calculate the number of samples in each segment
        samples_per_segment = int(segment_length * sr)

        # Calculate the number of segments
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
