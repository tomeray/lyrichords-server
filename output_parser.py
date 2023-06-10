class OutputProcessor:
    ids_to_chords = {}

    def get_id_of_max_value_and_convert_to_chord(self, chords_list: list, ids_to_chords: dict) -> list:
        chords_id_list = list(chords_list.argmax(axis=1))

        # Convert chord id list to chord name list
        chords_list = [ids_to_chords[chord_id] for chord_id in chords_id_list]
        return chords_list

    def parse_to_json(self, list_of_chords, chord_time_in_list) -> list:
        last_chord = None
        seconds_from = 0.0
        seconds_until = 0.0
        output = []

        for chord in list_of_chords:
            if not last_chord:
                seconds_until += chord_time_in_list
                last_chord = chord
            elif chord == last_chord:
                seconds_until += chord_time_in_list
            else:
                output.append({
                    'start': seconds_from,
                    'end': seconds_until,
                    'chord': last_chord
                })
                seconds_from = seconds_until
                seconds_until += chord_time_in_list
                last_chord = chord
        output.append({
            'start': seconds_from,
            'end': seconds_until,
            'chord': last_chord
        })
        return output

    def union_lists(self, *args):
        union_list = []
        for arg in args:
            union_list = union_list + arg
        return union_list

    def process_output(self, prediction):
        semitones = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        qualities = ["maj", "min", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "maj9", "7", "dim7",
                     "hdim7", "sus2", "sus4", "9", "min9", "min11", "maj11", "11", "13", "min13", "maj13"]
        special_chords = ["X", "N"]

        chord_names = []
        chord_to_ids = {}
        ids_to_chords = {}

        # Generate chord names and assign IDs
        chord_id = 0

        for semitone in semitones:
            chord_names.append(semitone)
            chord_to_ids[semitone] = chord_id
            ids_to_chords[chord_id] = semitone
            chord_id += 1
            for quality in qualities:
                chord_name = semitone + ":" + quality
                chord_names.append(chord_name)
                chord_to_ids[chord_name] = chord_id
                ids_to_chords[chord_id] = chord_name
                chord_id += 1

        # Add special chords
        for special_chord in special_chords:
            chord_names.append(special_chord)
            chord_to_ids[special_chord] = chord_id
            ids_to_chords[chord_id] = special_chord
            chord_id += 1

        CHORD_TIME = 10 / 749

        model_prediction_parts_chords_list = [self.get_id_of_max_value_and_convert_to_chord(pred, ids_to_chords) for pred in prediction]
        model_prediction_chords_list = self.union_lists(*model_prediction_parts_chords_list)
        return self.parse_to_json(model_prediction_chords_list, CHORD_TIME)