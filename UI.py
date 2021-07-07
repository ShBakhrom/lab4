# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
from flask import Flask, request, render_template



STATIC_DIR = os.path.abspath('static')

app=Flask(__name__)

pickle_in = open('MODEL_FOR_API.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''
    
    features = [x for x in request.form.values()]
    
    data = {'Weight': [features[0]],
        'Length1': [features[1]],
       'Length2': [features[2]],
       'Length3': [features[3]],
       'Height': [features[4]],
       'Width': [features[5]]}  

    mydf = pd.DataFrame(data)
    scaler2 = StandardScaler()
    scaler2=load('std_scaler.bin')
    mydf = pd.DataFrame(scaler2.transform(mydf), columns=mydf.columns)
    
    
    prediction = classifier.predict(mydf)[0]
    
    return render_template('index.html', prediction_text='It belongs to {} species'.format(str(prediction)))
    

if __name__=='__main__':
    app.run()