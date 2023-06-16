import os
import tempfile

import requests
import torch
from flask import Flask, request, stream_with_context, json, Response
from flask_cors import CORS, cross_origin
from pre_processor import PreProcessor
from output_parser import OutputProcessor

from ChordClassifier import ChordClassifier
from transformers import AutoModel
from io import BytesIO

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

pre_processor = PreProcessor()
output_parser = OutputProcessor()

NUM_CLASSES = 278
MODEL_NAME = "m-a-p/MERT-v1-330M"

mert_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
classifier_model = ChordClassifier(mert_model, 278)
classifier_model.load_state_dict(
    torch.load('trained_model/chord_classifier_model_good_save.pt', map_location=torch.device('cpu')))
classifier_model.eval()


def stream_model_results(preprocessed_segments, title, audio_file_link=None):
    yield '{'
    yield f'"title": "{title}", '
    if audio_file_link is not None:
        yield f'"audioUrl": "{audio_file_link}",'
    with torch.no_grad():
        yield '"chords": ['
        for index, processed_audio_data in enumerate(preprocessed_segments):
            output = output_parser.process_segment_output(classifier_model(**processed_audio_data)['predictions'],
                                                          index)
            print('yielding output ' + str(index))
            output_json_str = json.dumps(output)[1:-1]
            if index < len(preprocessed_segments) - 1:
                yield output_json_str + ','
            else:
                yield output_json_str
        yield ']'
    yield '}'


@app.route('/predict-audio', methods=['POST'])
@cross_origin()
def predict_audio():
    if 'file' not in request.files:
        return 'No file provided', 400

    file = request.files['file']

    if file.filename == '':
        return 'Empty file', 400

    preprocessed_segments = pre_processor.preprocess(file)
    return Response(stream_with_context(stream_model_results(preprocessed_segments, file.filename)),
                    content_type='application/json')


@app.route('/predict-youtube', methods=['GET'])
def predict_from_youtube_video_id():
    url = "https://youtube-mp36.p.rapidapi.com/dl"
    headers = {
        "X-RapidAPI-Key": "d4acfc805cmshea2037d8b8c7f5fp15e8f4jsnedd28f7a8217",
        "X-RapidAPI-Host": "youtube-mp36.p.rapidapi.com"
    }

    video_id = request.args.get("videoId")
    querystring = {"id": video_id}

    youtube_data_response = requests.get(url, headers=headers, params=querystring).json()
    print(youtube_data_response['link'])

    audio_file_response = requests.get(youtube_data_response['link'])
    audio_file_response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_file.write(audio_file_response.content)
        temp_file.close()
        file_path = temp_file.name

    preprocessed_segments = pre_processor.preprocess(file_path)
    os.remove(file_path)

    return Response(stream_with_context(stream_model_results(preprocessed_segments,
                                                             youtube_data_response['title'],
                                                             youtube_data_response['link'])),
                    content_type='application/json')
