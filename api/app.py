from flask import Flask, request, jsonify
import util
from flask_cors import CORS
import json
import numpy as np



# Create the Flask app object

app = Flask(__name__)
CORS(app)

spam_test_text = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"


@app.route('/handshake')
def index():
    # Perform prediction on the text input using the trained model
    prediction = util.predict(np.expand_dims(np.array(spam_test_text), axis = 0))

    if (prediction[0][0] == 'spam'):
        return jsonify({'test':1})
    else:
        return jsonify({'test':0})
    


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


