from flask import Flask, render_template , request, redirect
import os
from pickle import load
import numpy as np
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model, load_model
import base64
#---------------------------------------------------------------------#

app = Flask(__name__)

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def generate_feature(path):
    image = load_img(path, target_size = (224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length ):
    in_text = 'startseq'
    for i in range(max_length):
         sequence = tokenizer.texts_to_sequences([in_text])[0]
         sequence = pad_sequences([sequence], max_length)
         y = model.predict([image,sequence], verbose = 0)
         y = np.argmax(y)
         word = idx_to_word(y, tokenizer)
         if word is None:
             break
         in_text += " " + word

         if word == 'endseq':
             break
    return in_text

def generate_caption(path):
    features = generate_feature(path)
    tokenizer = load(open("tokenizer_best.p", "rb"))
    model = load_model('best_model.h5')
    predicted = predict_caption(model, features, tokenizer, 35)
    return predicted

def remove_start_end(caption, start_token, end_token):
    words = caption.split()
    if words[0] == start_token:
        words.pop(0)
    if words[-1] == end_token:
        words.pop(-1)
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('home.html')



#@app.route('/generated_caption')

@app.route('/upload' , methods = ['POST'])
def upload():
    file = request.files['file']
    extension = os.path.splitext(file.filename)
    if file:
        file.save(os.path.join('upload/', file.filename))
    #file_path = 'upload/' + file.filename
    file_path = 'upload/' + file.filename
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    caption = generate_caption(file_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    caption = remove_start_end(caption,'startseq','endseq')
    return render_template('generated_caption.html' , caption_to_print = caption, encoded_string = encoded_string)

if __name__ == '__main__':
    app.run(debug = True)