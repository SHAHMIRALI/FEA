import os
from flask import Flask, flash, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

def list_all_models():
    return 'Testing'

@app.route('/predict')
def get_user_image():
    return render_template('index.html')