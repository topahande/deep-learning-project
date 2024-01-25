import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

preprocessor = create_preprocessor('xception', target_size=(150, 150))

interpreter = tflite.Interpreter(model_path='fruit-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'Apple',
    'Banana',
    'Blueberry',
    'Lemon',
    'Orange',
    'Peach',
    'Pear',
    'Strawberry',
    'Tomato',
    'Watermelon'
]

app = Flask('fruit')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    X = preprocessor.from_url(data['path'])
    #X = preprocessor.from_path(data['path'])

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    result_dict = dict(zip(classes, float_predictions))

    result = max(result_dict, key=result_dict.get)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


