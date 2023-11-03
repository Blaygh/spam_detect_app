from flask import Flask, request, jsonify
import util
from flask_cors import CORS
import json



# Create the Flask app object

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    '''Performs prediction on the text input using the trained model.'''

    # Get the text input from the request
    data = json.loads(request.data)
    text = data.get('text')


    print(f'{text} formm') #debugging

    # Perform prediction on the text input using the trained model
    if text == '':
        return jsonify({'error': 'no text provided'}), 400
    prediction = util.predict([text])

    # Send back a response with the prediction
    response = jsonify({'prediction': prediction})

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


    
if __name__ == '__main__':
    util.load_data()
    util.vectorizer()
    app.run(debug=True)
