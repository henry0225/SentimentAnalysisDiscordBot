import random
from predictions import predict
import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.models.load_model('nn2')
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')

def get_response(message: str) -> str:
    p_message = message.lower()
    sentiment = model.predict(np.array([message]))[0][0]
    if sentiment > 0:
        return "That was a positive message with a rating of: " + str(sentiment)
    else:
        return "That was a negative message with a rating of: " + str(sentiment)