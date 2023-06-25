from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("best_model.h5")

@app.route('/')
def home():    
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def upload():
    if 'image' in request.files:
        image_file = request.files['image']
        image = Image.open(image_file)
        prediction =   classify_image(image)
        return render_template('index.html', prediction=prediction)

def classify_image(image):
    classes = ['benign','malignant']
    image = np.array(image,dtype=np.float64)
    image/=255
    image = cv2.resize(image,(224,224),cv2.INTER_AREA)
    pred = model.predict(np.expand_dims(image,axis=0))
    return classes[pred.argmax()]

if __name__ == '__main__':
    app.run()

