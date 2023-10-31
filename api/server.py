from flask import Flask, request, jsonify
import joblib
import util

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the request
    text = request.json['text']

    # Perform prediction on the text input using the trained model
    prediction = model.predict([text])[0]

    # Send back a response with the prediction
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
