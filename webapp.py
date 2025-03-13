import os
import cv2
import numpy as np
import speech_recognition as sr
from keras.models import model_from_json
from flask import Flask, render_template, jsonify, redirect, session, url_for, Response, request
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from pymongo import MongoClient
from transformers import pipeline
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'

bcrypt = Bcrypt(app)

# MongoDB Atlas Connection
client = MongoClient("mongodb+srv://<username>:<password>@cluster0.yjfhw6p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['EmotionDetection_login']
users = db['users']
results_collection = db['emotion_results']

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_id):
        self.id = str(user_id)

@login_manager.user_loader
def load_user(user_id):
    user = users.find_one({"_id": user_id})
    return User(user_id) if user else None

@app.route('/')
def home():
    return redirect(url_for('login'))

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
            session['username'] = username
            return redirect(url_for('base'))

        return "Invalid credentials!"

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/base')
@login_required
def base():
    return render_template('baseapp.html', username=session.get('username'))

@app.route('/start_analysis')
@login_required
def start_analysis():
    return redirect(url_for('realtime'))

@app.route('/realtime')
@login_required
def realtime():
    return render_template('realtimeanalysis.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_results')
@login_required
def view_results():
    username = session.get("username")
    results = list(results_collection.find({"username": username}, {"_id": 0}))
    return render_template('view_results.html', results=results)

# Load Emotion Detection Model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
latest_emotion = "Neutral"
latest_score = 0

# Initialize Audio Sentiment Analysis Model (BERT)
audio_sentiment_model = pipeline("sentiment-analysis")
latest_audio_text = "Listening..."
latest_audio_sentiment = "Neutral"
latest_audio_score = 0

def get_description(overall_sentiment, overall_score):
    """
    Generate description based on overall sentiment and score.
    If the sentiment is negative, add a doctor recommendation.
    """
    descriptions = {
        "happy": "Keep smiling! Happiness looks great on you.",
        "neutral": "You seem calm and balanced. Stay mindful.",
        "sad": "It's okay to feel down sometimes. Take care of yourself.",
        "angry": "Take a deep breath. Try to relax and clear your mind.",
        "fear": "Fear is natural. Stay calm and face your worries step by step.",
        "disgust": "Not everything is pleasant, but stay positive!",
        "surprise": "Exciting moments can be thrilling! Enjoy the surprises in life."
    }

    # Get the standard description
    description = descriptions.get(overall_sentiment.lower(), "Stay mindful and positive.")

    # If score is negative and emotion is in a depressed state, add doctor recommendation
    if overall_sentiment.lower() in ["sad", "fear", "angry", "disgust"] and overall_score < 0.4:
        description += " If you're feeling overwhelmed, consider talking to someone or seeking professional help."

    return description

def analyze_audio():
    global latest_audio_text, latest_audio_sentiment, latest_audio_score
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            latest_audio_text = recognizer.recognize_google(audio)
            sentiment_result = audio_sentiment_model(latest_audio_text)[0]
            latest_audio_sentiment = sentiment_result['label']
            latest_audio_score = round(float(sentiment_result['score']), 2)
        except:
            latest_audio_text = "Audio not clear."
            latest_audio_sentiment = "Neutral"
            latest_audio_score = 0.0

def generate_frames():
    global latest_emotion, latest_score
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = image.reshape(1, 48, 48, 1) / 255.0

            pred = model.predict(img)
            latest_emotion = labels[pred.argmax()]
            latest_score = round(float(pred.max()), 2)

            cv2.putText(frame, f"{latest_emotion} ({latest_score})", (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/get_emotion')
def get_emotion():
    analyze_audio()
    overall_sentiment = latest_emotion if latest_score >= latest_audio_score else latest_audio_sentiment
    overall_score = max(latest_score, latest_audio_score)
    return jsonify({
        "video_emotion": latest_emotion,
        "video_score": latest_score,
        "audio_emotion": latest_audio_sentiment,
        "audio_score": latest_audio_score,
        "overall_sentiment": overall_sentiment,
        "emotion_state": overall_sentiment,
        "description": get_description(overall_sentiment, overall_score)
    })

@app.route('/save_result', methods=['POST'])
@login_required
def save_result():
    data = request.json
    data["username"] = session.get("username")
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp
    results_collection.insert_one(data)  # Save result to MongoDB
    
    return jsonify({"message": "Result saved successfully!", "redirect": url_for('view_results')}), 200

if __name__ == '__main__':
    app.run(debug=True)
