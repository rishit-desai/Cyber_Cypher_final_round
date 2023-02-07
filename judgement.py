import tensorflow as tf
import pickle

model = tf.keras.models.load_model('judgement')
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=400)
    pred = model.predict(padded)
    return pred