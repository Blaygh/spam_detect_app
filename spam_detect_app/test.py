# from spam_detect_app.app import embed_text
# import tensorflow as tf
# import pickle

# predict  = tf.keras.models.load_model('trained_model')

# with open('test_data', 'rb') as f:
#     x_test = pickle.load(f)

# with open('spam_col_test_data', 'rb') as f:
#     y_test = pickle.load(f)

# y_hat = predict(embed_text(x_test))

# print ((y_hat == y_test).sum/len(y_test))


