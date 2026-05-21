import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')  # <-- This line avoids Tkinter GUI errors in web apps
import matplotlib.pyplot as plt

from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil


connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)
        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided, Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(os.path.join(dirPath, fileName))

        fileName = request.form['filename']
        dst = "static/images"

        def find_image_path(file_name):
            train_path = os.path.join("train", file_name)
            if os.path.exists(train_path):
                return train_path
            test_path = os.path.join("test", file_name)
            if os.path.exists(test_path):
                return test_path
            raise FileNotFoundError(f"{file_name} not found in train/ or test/ directories.")

        try:
            src = find_image_path(fileName)
            shutil.copy(src, dst)
            image = cv2.imread(src)
        except FileNotFoundError as e:
            return render_template('index.html', msg=str(e))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        retval2, threshold2 = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        cv2.imwrite('static/sharpened.jpg', sharpened)

        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'hyperspectral-{}-{}.model'.format(LR, '2conv-basic')

        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
            return verifying_data

        verify_data = process_verify_data()
        tf.compat.v1.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)
        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)
        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)
        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)
        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 5, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                             loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')

        fig = plt.figure()
        str_label = " "
        accuracy = ""
        pre = ""
        pre1 = ""

        for num, data in enumerate(verify_data):
            img_num = data[1]
            img_data = data[0]
            y = fig.add_subplot(3, 4, num + 1)
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            model_out = model.predict([data])[0]

            if np.argmax(model_out) == 0:
                str_label = "forest area"
                accuracy = "The predicted image of the hyperspectral image is forest areas with an accuracy of {}%".format(model_out[0] * 100)
                pre = "The information about this analysis are:\n\n"
                pre1 = [
                    "A forest area is a region covered primarily by trees and vegetation, forming a natural habitat for diverse wildlife.",
                    "Forests regulate climate and support water cycles.",
                    "They provide resources like timber, medicinal plants, and recreation.",
                    "They serve as carbon sinks to mitigate climate change."
                ]
            elif np.argmax(model_out) == 1:
                str_label = "water resources"
                accuracy = "The predicted image of the hyperspectral image is water resources with an accuracy of {}%".format(model_out[1] * 100)
                pre = "The information about this analysis are:\n\n"
                pre1 = [
                    "Water resources refer to natural sources of water used for various purposes.",
                    "These include surface water (rivers, lakes), groundwater (aquifers), and more.",
                    "Their sustainable management is crucial for human and ecological needs."
                ]
            elif np.argmax(model_out) == 2:
                str_label = "agricultural lands"
                accuracy = "The predicted image of the hyperspectral image is agricultural lands with an accuracy of {}%".format(model_out[2] * 100)
                pre = "The information about this analysis are:\n\n"
                pre1 = [
                    "Hyperspectral crop images capture detailed spectral information across wavelengths.",
                    "They help in analyzing crop health, stress, and nutrient content.",
                    "Supports applications in precision agriculture, disease detection, and yield optimization."
                ]
            elif np.argmax(model_out) == 3:
                str_label = "residential area"
                accuracy = "The predicted image of the hyperspectral image is residential area with an accuracy of {}%".format(model_out[3] * 100)
                pre = "The information about this analysis are:\n\n"
                pre1 = [
                    "A residential area is designated for housing and living purposes.",
                    "Includes homes, apartments, parks, schools, and public services.",
                    "Can range from low-density suburbs to high-density urban developments."
                ]
            elif np.argmax(model_out) == 4:
                str_label = "desert"
                accuracy = "The predicted image of the hyperspectral image is desert with an accuracy of {}%".format(model_out[4] * 100)
                pre = "The information about this analysis are:\n\n"
                pre1 = [
                    "A desert is a barren landscape receiving very little precipitation.",
                    "Typically less than 10 inches (25 cm) per year.",
                    "Deserts have sparse vegetation and harsh living conditions."
                ]

        return render_template('userlog.html',
                               status=str_label,
                               accuracy=accuracy,
                               Precaution=pre,
                               Precaution1=pre1,
                               ImageDisplay="http://127.0.0.1:5000/static/images/" + fileName,
                               ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",
                               ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",
                               ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",
                               ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
