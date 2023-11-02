import tensorflow as tf
import numpy as np
import pandas as pd



# load the trained model
# vectorizer function
# predict function

__trained_model = None
__spam_data = None
__textVectorizer = None


def load_data():
    '''returns the data as a pandas dataframe'''
    global __spam_data
    global __trained_model

    print ('...loading data')
    __spam_data = pd.read_csv('spam_detect_app/spam_preprocessed.csv')#development location is spam_detect_app/spam_preprocessed.csv
    print ('...loading model')
    __trained_model = tf.keras.models.load_model('trained_model')
    print('load successfull')
    

def vectorizer():
    '''Adapts vectorizer to create vocabulary and returns textVectorizer model with the adapted data'''
    global __textVectorizer

    __textVectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=100000, output_sequence_length=200)

    __textVectorizer.adapt(__spam_data.Message) #adapt data to create vocabulary

    return __textVectorizer



def predict(text):
    '''Performs prediction on the text input using the trained model.'''

    embedded_text = __textVectorizer(text) #vectorize data
    embedded_text = tf.expand_dims(embedded_text, -1)


    # Perform prediction on the text input using the trained model
    prediction = np.array(__trained_model.predict(embedded_text))
    
    prediction = np.where(prediction > 0.7, 'spam', 'ham')
    prediction = prediction.tolist()

    return prediction

if __name__ == '__main__':
    load_data()
    vectorizer()
    prediction = predict(['Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C\'s apply 08452810075over18\'s'])
    print(prediction)

# Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's


