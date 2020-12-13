import os
from flask import Flask, flash, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict')
def get_user_image():
    return render_template('index.html')