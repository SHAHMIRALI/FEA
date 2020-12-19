import os
from flask import Flask, flash, render_template, request, redirect, url_for

from common.database import list_data, auth_dropbox
from common.constants import IMAGE_UPLOAD_PATH, ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_PATH'] = IMAGE_UPLOAD_PATH
dbx = auth_dropbox()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/list')
def list_all_models():
    databases, test_pics = list_data(dbx)
    return {'databases': databases, 'test_pics': test_pics}

@app.route('/predict', methods=['GET', 'POST'])
def get_user_image():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
