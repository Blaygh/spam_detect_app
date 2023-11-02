 # Path: spam_detect_app/app.py

import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np

EPOCHS = 3

def embed_text_bert(spam_data):
    '''returns the pooled output of embedded input'''

    # bert modle not running due to missing files try to fix later,/
    # im going to embed data myself, using 
    # text vectotizer

    #bert api
    pre_process_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4' 

    bert_preprocess_model = hub.KerasLayer(pre_process_url)
    bert_model = hub.KerasLayer(encoder_url)


    preprocessed_text = bert_preprocess_model(spam_data)

    print ('...embedding data')

    embedding_result = bert_model(preprocessed_text)
    print('data embedding complete')

    return embedding_result['pooled_output']

def load_data():
    '''returns the data as a pandas dataframe'''
    print ('...loading data')
    spam_data = pd.read_csv('spam_detect_app/spam_preprocessed.csv')#development location is spam_detect_app/spam_preprocessed.csv
    print('data load successfull')
    return spam_data

def embed_text(spam_data):
    '''adapts data and returns textVectorizer model with the adapted data'''
    textVectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=100000, output_sequence_length=200)


    textVectorizer.adapt(spam_data) #adapt data to create vocabulary


    return textVectorizer


def vectorize_text(textVectorizer, spam_data):
    '''returns input text as an embedded text in numpy'''

    embedded_text = textVectorizer(spam_data) #vectorize data

    return np.array(embedded_text)

def pipe_data(x_data,y_data):
    '''returns a piped dataset(cached,shuffled,batched and prefetched) from the given data as a tuple of tensors'''
    test_data = tf.data.Dataset.from_tensor_slices((x_data,y_data))
    test_data = test_data.cache()
    test_data = test_data.shuffle(6000)
    test_data = test_data.batch(15)
    test_data = test_data.prefetch(5)

    return test_data


def model_create():
    '''returns a model with the given parameters'''
    input = tf.keras.layers.Input(shape=(200,))# Tensorflow autograph can race LSTM in a function, consider putting it in the main execution block
    embedding_layer = tf.keras.layers.Embedding(100000, 128)(input)
    lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True))(embedding_layer)
    flatten = tf.keras.layers.Flatten()(lstm_layer)
    hidden_layer1 = tf.keras.layers.Dense(32,activation= 'relu', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(flatten)  
    hidden_drop1 = tf.keras.layers.Dropout(0.2)(hidden_layer1)
    hidden_layer2 = tf.keras.layers.Dense(64,activation= 'relu',kernel_regularizer=tf.keras.regularizers.l1(0.001))(hidden_drop1)
    hidden_drop2 = tf.keras.layers.Dropout(0.5)(hidden_layer2)
    hidden_layer3 = tf.keras.layers.Dense(32,activation= 'relu', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(hidden_drop2)
    hidden_drop3 = tf.keras.layers.Dropout(0.2)(hidden_layer3)
    output = tf.keras.layers.Dense(1,activation= 'sigmoid')(hidden_drop3)

    model = tf.keras.models.Model(inputs = input, outputs = output)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model


if __name__ == '__main__':
    data = load_data()

    vectorizer = embed_text(data.Message)
    vectorized_data = vectorize_text(vectorizer,data.Message)

    ##split data into train and test samples
    x_train, x_test, y_train, y_test= train_test_split(vectorized_data,
                                                         data.spam,
                                                        test_size=0.1)
    
    #data pipeline
    test_data = pipe_data(x_train,np.expand_dims(y_train, axis=1))

    print(f'np.expand_dims(y_train, axis=1).shape:{np.expand_dims(y_train, axis=1).shape}')

    print(f'//////////////////////////TRAIN DATA///////////')



    model  = model_create()
    model.fit(test_data,epochs = EPOCHS)
    print('train done')

    print(f'//////////////////////////saving model///////////')
    tf.keras.models.save_model(model, 'trained_model')

    print(f'//////////////////////////saving model done///////////')

    print(f'//////////////////////////Saving Test data///////////')
    with open('test_data', 'wb') as f:#development location is spam_detect_app/test_data
        pickle.dump(x_test, f)

    with open('spam_col_test_data', 'wb') as f:#development location is spam_detect_app/spam_col_test_data
        pickle.dump(y_test,f)

    print(f'//////////////////////////Saving Test data done///////////')

    print(f'//////////////////////////Loading Trained model///////////')
    trained_model  = tf.keras.models.load_model('trained_model')


    print(f'//////////////////////////Loading Test Data///////////')
    with open('test_data', 'rb') as f:#development location is spam_detect_app/test_data
        x_test = pickle.load(f)
        print ('...test data load successfull')

    with open('spam_col_test_data', 'rb') as f:#development location is spam_detect_app/spam_col_test_data
        y_test = pickle.load(f)
        print ('...spam_col_test_data successfull')

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\test below
    # vectorized_data = vectorize_text(vectorizer,"WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.")
    # print(f'vectorized_data.shape:{vectorized_data.shape}')
    
    # print(f'np.expand_dims(vectorized_data, axis=0).shape: {np.expand_dims(vectorized_data, axis=0).shape}' )
    # y_hat = model.predict(np.expand_dims(vectorized_data, axis=0))

    # print(y_hat)
    # print(f"y_hat.shape: {y_hat.shape}")
    # # /print(f"y_hat[0][0]: {y_hat[0][0]}")               

    # y_hat = predict(x_test)
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\test above

    y_hat = model.predict(x_test)

    y_true = np.array(y_test)
    y_hat = np.array(y_hat)


    # print (y_true)
    print(y_hat.shape)
    print(y_true.shape)

    print(np.array(y_hat))
    print(y_true)
