from flask import Flask,render_template
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    '''return the web document html css js in the static folder: sitedocs'''
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)