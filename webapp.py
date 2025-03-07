import os
import cv2
from keras.models import model_from_json
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, redirect, session, url_for, Response
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from pymongo import MongoClient

# Define the upload folder and allowed video file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Needed for session management

bcrypt = Bcrypt(app)

# MongoDB Atlas Connection
client = MongoClient("mongodb+srv://<username>:<password>@cluster0.yjfhw6p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['EmotionDetection_login']
users = db['users']

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect unauthorized users to login

# User Model for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = str(user_id)  # Convert _id to string

@login_manager.user_loader
def load_user(user_id):
    user = users.find_one({"_id": user_id})
    if user:
        return User(user_id)
    return None

# Authentication Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

        if users.find_one({"_id": username}):
            return "User already exists!"

        users.insert_one({"_id": username, "password": hashed_pw})
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.find_one({"_id": username})

        if user and bcrypt.check_password_hash(user['password'], password):
            login_user(User(user["_id"]))
            session['username'] = username  # Store username in session
            return redirect(url_for('base'))  # Redirect to baseapp.html

        return "Invalid credentials!"

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('username', None)  # Remove session data
    return redirect(url_for('login'))

# Protecting Main Application Routes
@app.route('/')
@login_required  # Ensures only logged-in users can access
def index():
    return redirect(url_for('base'))  # Redirect to baseapp.html

@app.route('/base')
@login_required
def base():
    return render_template('baseapp.html', username=session.get('username'))

# Existing Code (No Changes)
def create_upload_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

create_upload_folder()

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def generate_frames():
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(frame, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request!'}), 400

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return jsonify({'message': 'No file selected!'}), 400

    if allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            uploaded_file.save(filepath)
            from main import main
            res_text, res_audio, res_visual, sentiment_result, sentiment_type = main(filepath)

            session['results'] = {
                'res_text': res_text,
                'res_audio': res_audio,
                'res_visual': res_visual,
                'sentiment_result': sentiment_result,
                'sentiment_type': sentiment_type
            }

            return redirect(url_for('show_results'))

        except Exception as e:
            return jsonify({'message': f"File upload or processing failed: {str(e)}"}), 500
    else:
        return jsonify({'message': 'File type not allowed!'}), 400

@app.route('/results')
@login_required
def show_results():
    results = session.get('results', None)
    if results:
        return render_template('results.html', results=results)
    else:
        return redirect(url_for('index'))

@app.route('/realtime')
@login_required
def realtime():
    return render_template('realtimeanalysis.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
