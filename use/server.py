import os
from flask import Flask, flash, render_template, request, redirect, url_for

from common.database import list_data, auth_dropbox

app = Flask(__name__)
dbx = auth_dropbox()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/list')
def list_all_models():
    return list_data(dbx)

@app.route('/predict')
def get_user_image():
    return render_template('index.html')