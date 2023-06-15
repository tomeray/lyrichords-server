import numpy as np


def union_lists(*args):
    union_list = []
    for arg in args:
        union_list = union_list + arg
    return union_list


def smooth_predictions(predictions, tolerance=20):

    def smooth_sequence(sequence):
        smoothed_sequence = []
        window_start = 0
        window_end = 0
        current_class = sequence[0]
        last_class = sequence[0]

        for i in range(1, len(sequence) + 1):
            # If there is a change in the current class being repeated
            if i == len(sequence) or sequence[i] != current_class:
                window_end = i
                length = window_end - window_start

                # If the current class has been repeated fewer times than the desired tolerance
                # We will replace it with the last class
                if length < tolerance:
                    smoothed_sequence.extend([last_class] * length)
                else:
                    last_class = current_class
                    smoothed_sequence.extend([current_class] * length)

                window_start = i
                current_class = sequence[i] if i < len(sequence) else current_class

        window_end = len(sequence)
        smoothed_sequence.extend([current_class] * (window_end - window_start))

        return smoothed_sequence

    # To smooth the results from both ends of the sequence
    for _ in range(2):
        predictions = smooth_sequence(predictions)[::-1]

    return np.array(predictions)


class OutputProcessor:
    
    def __init__(self):
        self.CHORD_TIME = 10 / 749

        semitones = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        qualities = ["min", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "maj9", "7", "dim7", "hdim7",
                     "sus2", "sus4", "9", "min9", "min11", "maj11", "11", "13", "min13", "maj13"]
        special_chords = ["X", "N"]

        chord_names = []
        self.chord_to_ids = {}
        self.ids_to_chords = {}

        # Generate chord names and assign IDs
        chord_id = 0

        for semitone in semitones:
            chord_names.append(semitone)
            self.chord_to_ids[semitone] = chord_id
            self.ids_to_chords[chord_id] = semitone
            chord_id += 1
            for quality in qualities:
                chord_name = semitone + ":" + quality
                chord_names.append(chord_name)
                self.chord_to_ids[chord_name] = chord_id
                self.ids_to_chords[chord_id] = chord_name
                chord_id += 1

        # Add special chords
        for special_chord in special_chords:
            chord_names.append(special_chord)
            self.chord_to_ids[special_chord] = chord_id
            self.ids_to_chords[chord_id] = special_chord
            chord_id += 1

        print(self.chord_to_ids["D#:11"])

    def get_predictions_as_chord_names(self, model_prediction) -> list:
        chords_id_list = smooth_predictions(model_prediction.argmax(-1).numpy()[0])
        print('original predictions')
        print(model_prediction.argmax(-1).numpy()[0])
        print('smoothed predictions')
        print(chords_id_list)

        # Convert chord id list to chord name list
        chords_names = [self.ids_to_chords[chord_id] for chord_id in chords_id_list]
        return chords_names

    def parse_to_json(self, list_of_chords, start_time) -> list:
        last_chord = None
        seconds_from = start_time
        seconds_until = start_time
        output = []

        for chord in list_of_chords:
            if not last_chord:
                seconds_until += self.CHORD_TIME
                last_chord = chord
            elif chord == last_chord:
                seconds_until += self.CHORD_TIME
            else:
                output.append({
                    'start': seconds_from,
                    'end': seconds_until,
                    'chord': last_chord
                })
                seconds_from = seconds_until
                seconds_until += self.CHORD_TIME
                last_chord = chord
        output.append({
            'start': seconds_from,
            'end': seconds_until,
            'chord': last_chord
        })
        return output

    def process_segment_output(self, prediction, index):
        chord_names_by_timestep = self.get_predictions_as_chord_names(prediction)
        return self.parse_to_json(chord_names_by_timestep, index * 10)

    def process_output(self, prediction):
        model_prediction_parts_chords_list = [self.get_predictions_as_chord_names(pred) for pred in prediction]
        model_prediction_chords_list = union_lists(*model_prediction_parts_chords_list)
        return self.parse_to_json(model_prediction_chords_list)