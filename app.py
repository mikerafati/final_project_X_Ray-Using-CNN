import os
from flask import Flask, redirect, url_for, request, jsonify, render_template
from flask_pymongo import PyMongo
import numpy as np
import keras
from keras.preprocessing import image
from keras import backend as K


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
mongo = PyMongo(app, uri="mongodb://localhost:27017/test_xray")

model = None
graph = None


# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
def load_model():
    global model
    global graph
    model = keras.models.load_model("Flask/balance_6epoch_vgg.h5")
    model._make_predict_function()
    graph = K.get_session().graph

load_model()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    # return render_template('index.html')
    pred_data = {}
    
    return render_template('index.html', pred=pred_data)

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():  
 
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file

            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)
            
            #load_model()
            preds = model_predict(filepath, model)
            print("Return Prediction is:")
            print(preds)
            max_preds = np.argmax(preds)
            print("Max preds is:" )
            print(max_preds)
           # In this model 1 is Pneumonia and 0 is Normal.
            prediction = ""
            str1 = 'Pneumonia'
            str2 = 'Normal'
            if max_preds == 1:
                prediction = str1
            else:
                prediction = str2
            img_path = filepath #'/uploads/{}'.format(filename)
            #label = prediction
            pred_data = {}
            pred_data['img_path'] = "../"+img_path
            pred_data['prediction'] = prediction

            print("img_path is")
            print(img_path)

            mongo.db.collection.update({}, pred_data, upsert=True)
            prediction_data = mongo.db.collection.find_one()
            print(prediction_data)
            return render_template('index.html', pred=prediction_data)
               

if __name__ == "__main__":
    app.run()









 


