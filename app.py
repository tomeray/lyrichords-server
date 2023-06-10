from flask import Flask, request
from pre_processor import PreProcessor
from output_parser import OutputProcessor
import numpy as np
import random

app = Flask(__name__)
pre_processor = PreProcessor()
output_parser = OutputProcessor()


# create demo prediction - REMOVE AFTER CONNECT MODEL
def create_model_prediction_demo(rows: int, cols: int) -> np.array:
    demo_model_prediction = np.empty((rows, cols))

    for i in range(rows):
        row_sum = 0
        for j in range(cols):
            demo_model_prediction[i][j] = random.random()
            row_sum += demo_model_prediction[i][j]

        for j in range(cols):
            demo_model_prediction[i][j] /= row_sum
    return demo_model_prediction


@app.route('/predict-audio', methods=['GET'])
def predict_audio():
    if 'file' not in request.files:
        return 'No file provided', 400

    file = request.files['file']

    if file.filename == '':
        return 'Empty file', 400

    # this will use to the model input
    segments = pre_processor.split_audio_into_segments(file)

    # demo prediction - REPLACE WITH REAL MODEL PREDICTION
    demo_model_prediction = [create_model_prediction_demo(749, 290)] * 5

    return output_parser.process_output(demo_model_prediction)
