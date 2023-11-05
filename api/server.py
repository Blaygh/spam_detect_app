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
    prediction = util.predict([text])

    # Send back a response with the prediction
    response = jsonify({'prediction': prediction})

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


# error handling

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({'error': 'not found'}), 404

# @app.errorhandler(500)
# def internal_server_error(error):
#     return jsonify({'error': 'internal server error'}), 500

# @app.errorhandler(400)
# def bad_request(error):
#     return jsonify({'error': 'bad request'}), 400
    
if __name__ == '__main__':
    util.load_data()
    util.vectorizer()
    app.run(debug=True)


