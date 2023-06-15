import torch
from flask import Flask, request, stream_with_context, json, Response
from flask_cors import CORS, cross_origin
from pre_processor import PreProcessor
from output_parser import OutputProcessor

from ChordClassifier import ChordClassifier
from transformers import AutoModel

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


@app.route('/predict-audio', methods=['POST'])
@cross_origin()
def predict_audio():
    if 'file' not in request.files:
        return 'No file provided', 400

    file = request.files['file']

    if file.filename == '':
        return 'Empty file', 400

    preprocessed_segments = pre_processor.preprocess(file)

    def stream_model_results():
        with torch.no_grad():
            yield '['
            for index, processed_audio_data in enumerate(preprocessed_segments):
                output = output_parser.process_segment_output(classifier_model(**processed_audio_data)['predictions'], index)
                print('yielding output ' + str(index))
                output_json_str = json.dumps(output)[1:-1]
                if index < len(preprocessed_segments) - 1:
                    yield output_json_str + ','
                else:
                    yield output_json_str
            yield ']'

    return Response(stream_with_context(stream_model_results()), content_type='application/json')
