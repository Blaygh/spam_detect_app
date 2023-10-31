import tensorflow as tf


def load_trained_model():
    '''Loads the trained model from the file system.'''
    trained_model  = tf.keras.models.load_model('trained_model')


