from flask import Flask, request, jsonify
import util
from flask_cors import CORS
import json
import numpy as np



# Create the Flask app object

app = Flask(__name__)
CORS(app)


@app.route('/handshake')
def index():
    # Perform prediction on the text input using the trained model
    prediction = util.predict(np.expand_dims(np.array('ham ham'), axis = 0))

    if (prediction == 'ham'):
        return jsonify({'test':'App and working'})
    else:
        return jsonify({'test':'System temporarily down'})


@app.route('/predict', methods=['POST'])
def predict():
    '''Performs prediction on the text input using the trained model.'''

# error handling
    # if request.method != 'POST':
    #     return jsonify({'error': 'use POST method'}), 405
    
    # if request.headers['Content-Type'] != 'application/json':
    #     return jsonify({'error': 'use JSON format'}), 415
    
    # if 'text' not in request.json:                                  
    #     return jsonify({'error': "no text provided. Make sure key is 'text' "}), 400
    
    # if len(request.json) > 1:
    #     return jsonify({'error': 'only one text input allowed'}), 400
    
    # if len(request.json['text']) > 1:
    #     print(request.json['text'])
    #     return jsonify({'error': 'only one text input allowed'}), 400
    
    # if len(request.json['text']) < 1:
    #     return jsonify({'error': 'no text provided'}), 400
    
    # if request.json['text'] == '':
    #     return jsonify({'error': 'no text provided'}), 400

    # Get the text input from the request
    data = json.loads(request.data)
    text = data.get('text')


    print(f'{text} formm') #debugging

    # Perform prediction on the text input using the trained model
    prediction = util.predict(np.expand_dims(np.array(text), axis = 0))

    # Send back a response with the prediction
    response = jsonify({'prediction': prediction})

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

 
if __name__ == '__main__':
    util.load_data()
    util.vectorizer()
    app.run(debug=True)


