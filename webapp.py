# import os
# import cv2
# import numpy as np
# import speech_recognition as sr
# from keras.models import model_from_json
# from flask import Flask, render_template, jsonify, redirect, session, url_for, Response, request
# from flask_bcrypt import Bcrypt
# from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
# from pymongo import MongoClient
# from transformers import pipeline
# from datetime import datetime
# import time
# from werkzeug.utils import secure_filename
# import tempfile
# from gtts import gTTS
# import base64
# from bson import ObjectId
# import wave
# import aifc
# import audioop
# from pydub import AudioSegment

# app = Flask(__name__)
# app.secret_key = 'supersecretkey'
# app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
# app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'aiff', 'flac'}

# bcrypt = Bcrypt(app)

# # MongoDB Atlas Connection
# client = MongoClient("mongodb+srv://rafath1234:rafath123@cluster0.yjfhw6p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# db = client['EmotionDetection_login']
# users = db['users']
# results_collection = db['emotion_result']
# sessions_collection = db['sessions']

# # Create indexes
# sessions_collection.create_index([("username", 1), ("timestamp", -1)])

# # Enhanced Therapy Content
# THERAPY_VIDEOS = {
#     "happy": "https://www.youtube.com/embed/1G4isv_Fylg",
#     "relax": "https://www.youtube.com/embed/m5TwT69i1lU",
#     "motivate": "https://www.youtube.com/embed/Rr5IjO_O3IY",
#     "calm": "https://www.youtube.com/embed/inpok4MKVLM",
#     "meditation": "https://www.youtube.com/embed/86m4RC_ADEY",
#     "anger": "https://www.youtube.com/embed/ZgH2y0Q6HsM",
#     "sadness": "https://www.youtube.com/embed/7jgrGSVHo4g",
#     "fear": "https://www.youtube.com/embed/8S4QBSgDn0w"
# }

# THERAPY_AUDIO = {
#     "calm": "static/audio/calm_music.mp3",
#     "meditation": "static/audio/guided_meditation.mp3",
#     "happy": "static/audio/happy_music.mp3",
#     "relax": "static/audio/relaxing_sounds.mp3",
#     "motivate": "static/audio/motivational_speech.mp3",
#     "anger": "static/audio/anger_management.mp3",
#     "sadness": "static/audio/sadness_relief.mp3",
#     "fear": "static/audio/fear_reduction.mp3"
# }

# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'

# class User(UserMixin):
#     def __init__(self, user_id):
#         self.id = str(user_id)

# @login_manager.user_loader
# def load_user(user_id):
#     user = users.find_one({"_id": user_id})
#     return User(user_id) if user else None

# # Load Emotion Detection Model
# json_file = open("emotiondetector.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("emotiondetector.h5")

# haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)

# labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
# latest_emotion = "neutral"
# latest_score = 0

# # Initialize NLP models
# audio_sentiment_model = pipeline("sentiment-analysis")
# text_sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
# latest_audio_text = "Listening..."
# latest_audio_sentiment = "neutral"
# latest_audio_score = 0

# def allowed_audio_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_EXTENSIONS']

# def convert_to_wav(input_path, output_path):
#     """Convert any audio file to WAV format"""
#     try:
#         audio = AudioSegment.from_file(input_path)
#         audio.export(output_path, format="wav")
#         return True
#     except Exception as e:
#         print(f"Error converting audio: {e}")
#         return False

# def is_valid_wav(filepath):
#     """Check if file is a valid WAV file"""
#     try:
#         with wave.open(filepath, 'rb') as wav_file:
#             return wav_file.getnframes() > 0
#     except:
#         return False

# @app.route('/')
# def home():
#     return redirect(url_for('login'))

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

#         if users.find_one({"_id": username}):
#             return "User already exists!"

#         users.insert_one({"_id": username, "password": hashed_pw})
#         return redirect(url_for('login'))

#     return render_template('signup.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         user = users.find_one({"_id": username})

#         if user and bcrypt.check_password_hash(user['password'], password):
#             login_user(User(user["_id"]))
#             session['username'] = username
#             return redirect(url_for('base'))

#         return "Invalid credentials!"

#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     session.pop('username', None)
#     return redirect(url_for('login'))

# @app.route('/base')
# @login_required
# def base():
#     return render_template('baseapp.html', username=session.get('username'))

# @app.route('/start_analysis')
# @login_required
# def start_analysis():
#     return redirect(url_for('realtime'))

# @app.route('/realtime')
# @login_required
# def realtime():
#     return render_template('realtimeanalysis.html')

# @app.route('/video_feed')
# @login_required
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/analyze_text', methods=['POST'])
# @login_required
# def analyze_text():
#     text = request.json.get('text', '')
    
#     try:
#         result = text_sentiment_model(text)[0]
#         sentiment = result['label'].lower()
#         score = round(float(result['score']), 2)
        
#         audio_response = generate_audio_response(text, sentiment, score)
        
#         global latest_audio_text, latest_audio_sentiment, latest_audio_score
#         latest_audio_text = text
#         latest_audio_sentiment = sentiment
#         latest_audio_score = score
        
#         session_id = sessions_collection.find_one(
#             {"username": session.get("username")},
#             sort=[("timestamp", -1)]
#         )["_id"]
        
#         sessions_collection.update_one(
#             {"_id": session_id},
#             {"$push": {"chat_history": {
#                 "text": text,
#                 "sender": "user",
#                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }}}
#         )
        
#         sessions_collection.update_one(
#             {"_id": session_id},
#             {"$push": {"chat_history": {
#                 "text": f"Analysis Result: {sentiment} (Score: {score})",
#                 "sender": "bot",
#                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "audio_response": audio_response
#             }}}
#         )
        
#         return jsonify({
#             'emotion': sentiment,
#             'score': score,
#             'audio_response': audio_response
#         })
#     except Exception as e:
#         print(f"Error analyzing text: {e}")
#         return jsonify({
#             'emotion': 'neutral',
#             'score': 0.5,
#             'audio_response': None
#         })

# @app.route('/analyze_audio', methods=['POST'])
# @login_required
# def analyze_audio():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'}), 400
    
#     audio_file = request.files['audio']
    
#     if audio_file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if not allowed_audio_file(audio_file.filename):
#         return jsonify({'error': 'File type not allowed'}), 400
    
#     try:
#         # Create temp files
#         temp_dir = tempfile.mkdtemp()
#         original_path = os.path.join(temp_dir, secure_filename(audio_file.filename))
#         wav_path = os.path.join(temp_dir, "converted.wav")
        
#         # Save original file
#         audio_file.save(original_path)
        
#         # Convert to WAV if needed
#         if not original_path.lower().endswith('.wav') or not is_valid_wav(original_path):
#             if not convert_to_wav(original_path, wav_path):
#                 raise ValueError("Audio conversion failed")
#             audio_path = wav_path
#         else:
#             audio_path = original_path
        
#         # Process audio
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(audio_path) as source:
#             audio = recognizer.record(source)
#             text = recognizer.recognize_google(audio)
            
#             result = text_sentiment_model(text)[0]
#             sentiment = result['label'].lower()
#             score = round(float(result['score']), 2)
            
#             audio_response = generate_audio_response(text, sentiment, score)
            
#             global latest_audio_text, latest_audio_sentiment, latest_audio_score
#             latest_audio_text = text
#             latest_audio_sentiment = sentiment
#             latest_audio_score = score
            
#             session_id = sessions_collection.find_one(
#                 {"username": session.get("username")},
#                 sort=[("timestamp", -1)]
#             )["_id"]
            
#             sessions_collection.update_one(
#                 {"_id": session_id},
#                 {"$push": {"chat_history": {
#                     "text": "Audio message: " + text,
#                     "sender": "user",
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 }}}
#             )
            
#             sessions_collection.update_one(
#                 {"_id": session_id},
#                 {"$push": {"chat_history": {
#                     "text": f"Analysis Result: {sentiment} (Score: {score})",
#                     "sender": "bot",
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "audio_response": audio_response
#                 }}}
#             )
            
#             return jsonify({
#                 'text': text,
#                 'emotion': sentiment,
#                 'score': score,
#                 'audio_response': audio_response
#             })
            
#     except sr.UnknownValueError:
#         return jsonify({
#             'text': "Could not understand audio",
#             'emotion': 'neutral',
#             'score': 0.5,
#             'audio_response': None
#         }), 400
#     except sr.RequestError as e:
#         return jsonify({
#             'error': f"Could not request results from Google Speech Recognition service: {e}",
#             'text': "",
#             'emotion': 'neutral',
#             'score': 0.5,
#             'audio_response': None
#         }), 500
#     except Exception as e:
#         print(f"Error analyzing audio: {e}")
#         return jsonify({
#             'error': str(e),
#             'text': "",
#             'emotion': 'neutral',
#             'score': 0.5,
#             'audio_response': None
#         }), 500
#     finally:
#         # Clean up temp files
#         try:
#             if 'original_path' in locals() and os.path.exists(original_path):
#                 os.unlink(original_path)
#             if 'wav_path' in locals() and os.path.exists(wav_path):
#                 os.unlink(wav_path)
#             if 'temp_dir' in locals() and os.path.exists(temp_dir):
#                 os.rmdir(temp_dir)
#         except Exception as e:
#             print(f"Error cleaning up temp files: {e}")

# def generate_audio_response(text, emotion, score):
#     try:
#         response_text = f"The analysis shows {emotion} sentiment with a score of {score}. "
        
#         if emotion in ['sad', 'angry', 'fear']:
#             response_text += "I notice you might be feeling down. Here's some content to help you feel better."
#         else:
#             response_text += "That's great! Keep up the positive mood."
        
#         tts = gTTS(text=response_text, lang='en')
#         temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
#         tts.save(temp_audio.name)
        
#         with open(temp_audio.name, 'rb') as f:
#             audio_data = f.read()
#         os.unlink(temp_audio.name)
        
#         return f"data:audio/mp3;base64,{base64.b64encode(audio_data).decode('utf-8')}"
#     except Exception as e:
#         print(f"Error generating audio response: {e}")
#         return None

# @app.route('/get_emotion')
# @login_required
# def get_emotion():
#     # Determine overall sentiment
#     if latest_score == 0 and latest_audio_score == 0:
#         overall_sentiment = "neutral"
#         overall_score = 0.5
#     else:
#         overall_sentiment = latest_emotion if latest_score >= latest_audio_score else latest_audio_sentiment
#         overall_score = max(latest_score, latest_audio_score)
    
#     # Determine if therapy should be shown
#     show_therapy = False
#     negative_emotions = ['sad', 'angry', 'fear', 'disgust']
    
#     # Check both video and audio for negative emotions with significant scores
#     video_needs_therapy = latest_emotion.lower() in negative_emotions and latest_score > 0.6
#     audio_needs_therapy = latest_audio_sentiment.lower() in negative_emotions and latest_audio_score > 0.6
    
#     show_therapy = video_needs_therapy or audio_needs_therapy
    
#     # Select appropriate therapy content
#     therapy_video = ""
#     therapy_audio = ""
#     therapy_message = ""
    
#     if show_therapy:
#         # For video-based negative emotions
#         if video_needs_therapy:
#             if latest_emotion.lower() == 'angry':
#                 therapy_video = THERAPY_VIDEOS["anger"]
#                 therapy_audio = THERAPY_AUDIO["anger"]
#             elif latest_emotion.lower() == 'sad':
#                 therapy_video = THERAPY_VIDEOS["sadness"]
#                 therapy_audio = THERAPY_AUDIO["sadness"]
#             elif latest_emotion.lower() == 'fear':
#                 therapy_video = THERAPY_VIDEOS["fear"]
#                 therapy_audio = THERAPY_AUDIO["fear"]
#             elif latest_emotion.lower() == 'disgust':
#                 therapy_video = THERAPY_VIDEOS["relax"]
#                 therapy_audio = THERAPY_AUDIO["relax"]
        
#         # For audio-based negative emotions
#         if audio_needs_therapy:
#             if latest_audio_sentiment.lower() == 'angry':
#                 therapy_audio = THERAPY_AUDIO["anger"] if not therapy_audio else therapy_audio
#                 therapy_video = THERAPY_VIDEOS["anger"] if not therapy_video else therapy_video
#             elif latest_audio_sentiment.lower() == 'sad':
#                 therapy_audio = THERAPY_AUDIO["sadness"] if not therapy_audio else therapy_audio
#                 therapy_video = THERAPY_VIDEOS["sadness"] if not therapy_video else therapy_video
#             elif latest_audio_sentiment.lower() == 'fear':
#                 therapy_audio = THERAPY_AUDIO["fear"] if not therapy_audio else therapy_audio
#                 therapy_video = THERAPY_VIDEOS["fear"] if not therapy_video else therapy_video
#             elif latest_audio_sentiment.lower() == 'disgust':
#                 therapy_audio = THERAPY_AUDIO["relax"] if not therapy_audio else therapy_audio
#                 therapy_video = THERAPY_VIDEOS["relax"] if not therapy_video else therapy_video
        
#         therapy_message = "Based on your current emotional state, we recommend the following content to help improve your mood."
    
#     description = get_description(overall_sentiment, overall_score)
    
#     return jsonify({
#         "video_emotion": latest_emotion,
#         "video_score": latest_score,
#         "audio_emotion": latest_audio_sentiment,
#         "audio_score": latest_audio_score,
#         "overall_sentiment": overall_sentiment,
#         "emotion_state": overall_sentiment,
#         "description": description,
#         "show_therapy": show_therapy,
#         "therapy_video": therapy_video,
#         "therapy_audio": therapy_audio,
#         "therapy_message": therapy_message
#     })

# @app.route('/save_result', methods=['POST'])
# @login_required
# def save_result():
#     data = request.json
#     data["username"] = session.get("username")
#     data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     data["session_duration"] = data.get("session_duration", 0)
#     data["session_ended_manually"] = data.get("session_ended_manually", False)
    
#     session_id = sessions_collection.insert_one(data).inserted_id
    
#     return jsonify({
#         "message": "Result saved successfully!", 
#         "session_id": str(session_id)
#     }), 200

# @app.route('/end_session', methods=['POST'])
# @login_required
# def end_session():
#     try:
#         # Get current session data
#         response = request.get_json()
#         if not response:
#             response = {}
        
#         # Add session metadata
#         response["username"] = session.get("username")
#         response["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         response["session_ended_manually"] = True
        
#         # Save to database
#         session_id = sessions_collection.insert_one(response).inserted_id
        
#         return jsonify({
#             "status": "success",
#             "message": "Session ended successfully",
#             "session_id": str(session_id)
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/therapy_engaged', methods=['POST'])
# @login_required
# def therapy_engaged():
#     session_id = sessions_collection.find_one(
#         {"username": session.get("username")},
#         sort=[("timestamp", -1)]
#     )["_id"]
    
#     sessions_collection.update_one(
#         {"_id": session_id},
#         {"$set": {"therapy_engaged": True}}
#     )
#     return jsonify({"status": "success"})

# @app.route('/get_history')
# @login_required
# def get_history():
#     username = session.get("username")
#     history = list(sessions_collection.find(
#         {"username": username},
#         {
#             "_id": 1,
#             "timestamp": 1,
#             "overall_sentiment": 1,
#             "description": 1,
#             "session_duration": 1,
#             "session_ended_manually": 1
#         }
#     ).sort("timestamp", -1).limit(10))
    
#     for item in history:
#         item["_id"] = str(item["_id"])
#         item.setdefault("overall_sentiment", "Unknown")
#         item.setdefault("description", "No description available")
#         item.setdefault("session_duration", 0)
#         item.setdefault("session_ended_manually", False)
    
#     return jsonify(history)

# @app.route('/view_results')
# @login_required
# def view_results():
#     username = session.get("username")
#     results = list(sessions_collection.find(
#         {"username": username},
#         {
#             "_id": 1,
#             "timestamp": 1,
#             "overall_sentiment": 1,
#             "description": 1,
#             "session_duration": 1,
#             "session_ended_manually": 1
#         }
#     ).sort("timestamp", -1))
    
#     # Process results to ensure consistent structure
#     processed_results = []
#     for result in results:
#         processed = {
#             "_id": str(result["_id"]),
#             "timestamp": result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
#             "overall_sentiment": result.get("overall_sentiment", "Unknown"),
#             "description": result.get("description", "No description available"),
#             "session_duration": result.get("session_duration", 0),
#             "session_ended_manually": result.get("session_ended_manually", False)
#         }
#         processed_results.append(processed)
    
#     return render_template('view_results.html', results=processed_results)

# @app.route('/session_details/<session_id>')
# @login_required
# def session_details(session_id):
#     session_data = sessions_collection.find_one({"_id": ObjectId(session_id)})
#     if not session_data:
#         return "Session not found", 404
    
#     session_data["_id"] = str(session_data["_id"])
#     return render_template('session_details.html', session=session_data)

# def get_description(overall_sentiment, overall_score):
#     descriptions = {
#         "happy": "Keep smiling! Happiness looks great on you.",
#         "neutral": "You seem calm and balanced. Stay mindful.",
#         "sad": "It's okay to feel down sometimes. Take care of yourself.",
#         "angry": "Take a deep breath. Try to relax and clear your mind.",
#         "fear": "Fear is natural. Stay calm and face your worries step by step.",
#         "disgust": "Not everything is pleasant, but stay positive!",
#         "surprise": "Exciting moments can be thrilling! Enjoy the surprises in life."
#     }

#     description = descriptions.get(overall_sentiment.lower(), "Stay mindful and positive.")

#     if overall_sentiment.lower() in ["sad", "fear", "angry", "disgust"] and overall_score < 0.4:
#         description += " If you're feeling overwhelmed, consider talking to someone or seeking professional help."

#     return description

# def generate_frames():
#     global latest_emotion, latest_score
#     webcam = cv2.VideoCapture(0)
#     while True:
#         success, frame = webcam.read()
#         if not success:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (p, q, r, s) in faces:
#             image = gray[q:q+s, p:p+r]
#             cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
#             image = cv2.resize(image, (48, 48))
#             img = image.reshape(1, 48, 48, 1) / 255.0

#             pred = model.predict(img)
#             latest_emotion = labels[pred.argmax()]
#             latest_score = round(float(pred.max()), 2)

#             cv2.putText(frame, f"{latest_emotion} ({latest_score})", (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# if __name__ == '__main__':
#     app.run(debug=True)

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
import time
from werkzeug.utils import secure_filename
import tempfile
from gtts import gTTS
import base64
from bson import ObjectId
import wave
import aifc
import audioop
from pydub import AudioSegment
import io

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'aiff', 'flac'}

bcrypt = Bcrypt(app)

# MongoDB Atlas Connection
client = MongoClient("mongodb+srv://<username>:<password>@cluster0.yjfhw6p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['EmotionDetection_login']
users = db['users']
results_collection = db['emotion_result']
sessions_collection = db['sessions']

# Create indexes
sessions_collection.create_index([("username", 1), ("timestamp", -1)])

# Enhanced Therapy Content
THERAPY_VIDEOS = {
    "happy": "https://www.youtube.com/embed/1G4isv_Fylg",
    "relax": "https://www.youtube.com/embed/m5TwT69i1lU",
    "motivate": "https://www.youtube.com/embed/Rr5IjO_O3IY",
    "calm": "https://www.youtube.com/embed/inpok4MKVLM",
    "meditation": "https://www.youtube.com/embed/86m4RC_ADEY",
    "anger": "https://www.youtube.com/embed/ZgH2y0Q6HsM",
    "sadness": "https://www.youtube.com/embed/7jgrGSVHo4g",
    "fear": "https://www.youtube.com/embed/8S4QBSgDn0w"
}

THERAPY_AUDIO = {
    "calm": "static/audio/calm_music.mp3",
    "meditation": "static/audio/guided_meditation.mp3",
    "happy": "static/audio/happy_music.mp3",
    "relax": "static/audio/relaxing_sounds.mp3",
    "motivate": "static/audio/motivational_speech.mp3",
    "anger": "static/audio/anger_management.mp3",
    "sadness": "static/audio/sadness_relief.mp3",
    "fear": "static/audio/fear_reduction.mp3"
}

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

# Load Emotion Detection Model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
latest_emotion = "neutral"
latest_score = 0

# Initialize NLP models
audio_sentiment_model = pipeline("sentiment-analysis")
text_sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
latest_audio_text = "Listening..."
latest_audio_sentiment = "neutral"
latest_audio_score = 0

def allowed_audio_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_EXTENSIONS']

def convert_to_wav(input_path, output_path):
    """Convert any audio file to WAV format"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False

def is_valid_wav(filepath):
    """Check if file is a valid WAV file"""
    try:
        with wave.open(filepath, 'rb') as wav_file:
            return wav_file.getnframes() > 0
    except:
        return False

def generate_audio_response(text, emotion, score):
    try:
        # Create more natural response text
        response_text = ""
        if emotion in ['sad', 'angry', 'fear', 'disgust']:
            response_text = f"I can sense you're feeling {emotion}. It's okay to feel this way sometimes. "
            response_text += "Here are some recommendations that might help you feel better."
        else:
            response_text = f"I notice you're feeling {emotion}. That's great! "
            response_text += "Would you like to share more about what's making you feel this way?"
        
        # Generate speech
        tts = gTTS(text=response_text, lang='en', slow=False)
        
        # Create in-memory file
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        audio_buffer.close()
        
        return f"data:audio/mp3;base64,{audio_base64}"
        
    except Exception as e:
        print(f"Error generating audio response: {e}")
        return None

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
            # Create a new session record
            sessions_collection.insert_one({
                "username": username,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_start": datetime.now(),
                "chat_history": []
            })
            return redirect(url_for('base'))

        return "Invalid credentials!"

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    username = session.get('username')
    if username:
        # First find the most recent session for this user
        recent_session = sessions_collection.find_one(
            {"username": username},
            sort=[("timestamp", -1)]
        )
        
        # If found, update it
        if recent_session:
            sessions_collection.update_one(
                {"_id": recent_session["_id"]},
                {"$set": {"session_end": datetime.now()}}
            )
    
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

@app.route('/analyze_text', methods=['POST'])
@login_required
def analyze_text():
    text = request.json.get('text', '')
    
    try:
        result = text_sentiment_model(text)[0]
        sentiment = result['label'].lower()
        score = round(float(result['score']), 2)
        
        audio_response = generate_audio_response(text, sentiment, score)
        
        global latest_audio_text, latest_audio_sentiment, latest_audio_score
        latest_audio_text = text
        latest_audio_sentiment = sentiment
        latest_audio_score = score
        
        session_id = sessions_collection.find_one(
            {"username": session.get("username")},
            sort=[("timestamp", -1)]
        )["_id"]
        
        sessions_collection.update_one(
            {"_id": session_id},
            {"$push": {"chat_history": {
                "text": text,
                "sender": "user",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }}}
        )
        
        sessions_collection.update_one(
            {"_id": session_id},
            {"$push": {"chat_history": {
                "text": f"Analysis Result: {sentiment} (Score: {score})",
                "sender": "bot",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audio_response": audio_response
            }}}
        )
        
        return jsonify({
            'emotion': sentiment,
            'score': score,
            'audio_response': audio_response
        })
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return jsonify({
            'emotion': 'neutral',
            'score': 0.5,
            'audio_response': None
        })

@app.route('/analyze_audio', methods=['POST'])
@login_required
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_audio_file(audio_file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Create temp files
        temp_dir = tempfile.mkdtemp()
        original_path = os.path.join(temp_dir, secure_filename(audio_file.filename))
        wav_path = os.path.join(temp_dir, "converted.wav")
        
        # Save original file
        audio_file.save(original_path)
        
        # Convert to WAV if needed
        if not original_path.lower().endswith('.wav') or not is_valid_wav(original_path):
            if not convert_to_wav(original_path, wav_path):
                raise ValueError("Audio conversion failed")
            audio_path = wav_path
        else:
            audio_path = original_path
        
        # Process audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            
            result = text_sentiment_model(text)[0]
            sentiment = result['label'].lower()
            score = round(float(result['score']), 2)
            
            audio_response = generate_audio_response(text, sentiment, score)
            
            global latest_audio_text, latest_audio_sentiment, latest_audio_score
            latest_audio_text = text
            latest_audio_sentiment = sentiment
            latest_audio_score = score
            
            session_id = sessions_collection.find_one(
                {"username": session.get("username")},
                sort=[("timestamp", -1)]
            )["_id"]
            
            sessions_collection.update_one(
                {"_id": session_id},
                {"$push": {"chat_history": {
                    "text": "Audio message: " + text,
                    "sender": "user",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }}}
            )
            
            sessions_collection.update_one(
                {"_id": session_id},
                {"$push": {"chat_history": {
                    "text": f"Analysis Result: {sentiment} (Score: {score})",
                    "sender": "bot",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "audio_response": audio_response
                }}}
            )
            
            return jsonify({
                'text': text,
                'emotion': sentiment,
                'score': score,
                'audio_response': audio_response
            })
            
    except sr.UnknownValueError:
        return jsonify({
            'text': "Could not understand audio",
            'emotion': 'neutral',
            'score': 0.5,
            'audio_response': None
        }), 400
    except sr.RequestError as e:
        return jsonify({
            'error': f"Could not request results from Google Speech Recognition service: {e}",
            'text': "",
            'emotion': 'neutral',
            'score': 0.5,
            'audio_response': None
        }), 500
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return jsonify({
            'error': str(e),
            'text': "",
            'emotion': 'neutral',
            'score': 0.5,
            'audio_response': None
        }), 500
    finally:
        # Clean up temp files
        try:
            if 'original_path' in locals() and os.path.exists(original_path):
                os.unlink(original_path)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.unlink(wav_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

@app.route('/get_emotion')
@login_required
def get_emotion():
    # Determine overall sentiment
    if latest_score == 0 and latest_audio_score == 0:
        overall_sentiment = "neutral"
        overall_score = 0.5
    else:
        overall_sentiment = latest_emotion if latest_score >= latest_audio_score else latest_audio_sentiment
        overall_score = max(latest_score, latest_audio_score)
    
    # Determine if therapy should be shown
    show_therapy = False
    negative_emotions = ['sad', 'angry', 'fear', 'disgust']
    
    # Check both video and audio for negative emotions with significant scores
    video_needs_therapy = latest_emotion.lower() in negative_emotions and latest_score > 0.6
    audio_needs_therapy = latest_audio_sentiment.lower() in negative_emotions and latest_audio_score > 0.6
    
    show_therapy = video_needs_therapy or audio_needs_therapy
    
    # Select appropriate therapy content
    therapy_video = ""
    therapy_audio = ""
    therapy_message = ""
    
    if show_therapy:
        # For video-based negative emotions
        if video_needs_therapy:
            if latest_emotion.lower() == 'angry':
                therapy_video = THERAPY_VIDEOS["anger"]
                therapy_audio = THERAPY_AUDIO["anger"]
            elif latest_emotion.lower() == 'sad':
                therapy_video = THERAPY_VIDEOS["sadness"]
                therapy_audio = THERAPY_AUDIO["sadness"]
            elif latest_emotion.lower() == 'fear':
                therapy_video = THERAPY_VIDEOS["fear"]
                therapy_audio = THERAPY_AUDIO["fear"]
            elif latest_emotion.lower() == 'disgust':
                therapy_video = THERAPY_VIDEOS["relax"]
                therapy_audio = THERAPY_AUDIO["relax"]
        
        # For audio-based negative emotions
        if audio_needs_therapy:
            if latest_audio_sentiment.lower() == 'angry':
                therapy_audio = THERAPY_AUDIO["anger"] if not therapy_audio else therapy_audio
                therapy_video = THERAPY_VIDEOS["anger"] if not therapy_video else therapy_video
            elif latest_audio_sentiment.lower() == 'sad':
                therapy_audio = THERAPY_AUDIO["sadness"] if not therapy_audio else therapy_audio
                therapy_video = THERAPY_VIDEOS["sadness"] if not therapy_video else therapy_video
            elif latest_audio_sentiment.lower() == 'fear':
                therapy_audio = THERAPY_AUDIO["fear"] if not therapy_audio else therapy_audio
                therapy_video = THERAPY_VIDEOS["fear"] if not therapy_video else therapy_video
            elif latest_audio_sentiment.lower() == 'disgust':
                therapy_audio = THERAPY_AUDIO["relax"] if not therapy_audio else therapy_audio
                therapy_video = THERAPY_VIDEOS["relax"] if not therapy_video else therapy_video
        
        therapy_message = "Based on your current emotional state, we recommend the following content to help improve your mood."
    
    description = get_description(overall_sentiment, overall_score)
    
    return jsonify({
        "video_emotion": latest_emotion,
        "video_score": latest_score,
        "audio_emotion": latest_audio_sentiment,
        "audio_score": latest_audio_score,
        "overall_sentiment": overall_sentiment,
        "emotion_state": overall_sentiment,
        "description": description,
        "show_therapy": show_therapy,
        "therapy_video": therapy_video,
        "therapy_audio": therapy_audio,
        "therapy_message": therapy_message
    })

@app.route('/save_result', methods=['POST'])
@login_required
def save_result():
    data = request.json
    data["username"] = session.get("username")
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["session_duration"] = data.get("session_duration", 0)
    data["session_ended_manually"] = data.get("session_ended_manually", False)
    
    session_id = sessions_collection.insert_one(data).inserted_id
    
    return jsonify({
        "message": "Result saved successfully!", 
        "session_id": str(session_id)
    }), 200

@app.route('/end_session', methods=['POST'])
@login_required
def end_session():
    try:
        # Get current session data
        response = request.get_json()
        if not response:
            response = {}
        
        # Add session metadata
        response["username"] = session.get("username")
        response["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response["session_ended_manually"] = True
        
        # Save to database
        session_id = sessions_collection.insert_one(response).inserted_id
        
        return jsonify({
            "status": "success",
            "message": "Session ended successfully",
            "session_id": str(session_id)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/therapy_engaged', methods=['POST'])
@login_required
def therapy_engaged():
    session_id = sessions_collection.find_one(
        {"username": session.get("username")},
        sort=[("timestamp", -1)]
    )["_id"]
    
    sessions_collection.update_one(
        {"_id": session_id},
        {"$set": {"therapy_engaged": True}}
    )
    return jsonify({"status": "success"})

@app.route('/get_history')
@login_required
def get_history():
    username = session.get("username")
    history = list(sessions_collection.find(
        {"username": username},
        {
            "_id": 1,
            "timestamp": 1,
            "overall_sentiment": 1,
            "description": 1,
            "session_duration": 1,
            "session_ended_manually": 1,
            "chat_history": {"$slice": -3}  # Get last 3 messages
        }
    ).sort("timestamp", -1).limit(10))
    
    for item in history:
        item["_id"] = str(item["_id"])
        item.setdefault("overall_sentiment", "Unknown")
        item.setdefault("description", "No description available")
        item.setdefault("session_duration", 0)
        item.setdefault("session_ended_manually", False)
        item.setdefault("chat_history", [])
    
    return jsonify(history)

@app.route('/view_results')
@login_required
def view_results():
    username = session.get("username")
    results = list(sessions_collection.find(
        {"username": username},
        {
            "_id": 1,
            "timestamp": 1,
            "overall_sentiment": 1,
            "description": 1,
            "session_duration": 1,
            "session_ended_manually": 1,
            "chat_history": {"$slice": -3}  # Get last 3 messages
        }
    ).sort("timestamp", -1))
    
    # Process results to ensure consistent structure
    processed_results = []
    for result in results:
        processed = {
            "_id": str(result["_id"]),
            "timestamp": result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "overall_sentiment": result.get("overall_sentiment", "Unknown"),
            "description": result.get("description", "No description available"),
            "session_duration": result.get("session_duration", 0),
            "session_ended_manually": result.get("session_ended_manually", False),
            "chat_history": result.get("chat_history", [])
        }
        processed_results.append(processed)
    
    return render_template('view_results.html', results=processed_results)

@app.route('/session_details/<session_id>')
@login_required
def session_details(session_id):
    try:
        session_data = sessions_collection.find_one({"_id": ObjectId(session_id)})
        if not session_data:
            return "Session not found", 404
        
        # Ensure all required fields exist with defaults if missing
        session_data.setdefault("video_emotion", "Unknown")
        session_data.setdefault("video_score", 0)
        session_data.setdefault("audio_emotion", "Unknown")
        session_data.setdefault("audio_score", 0)
        session_data.setdefault("overall_sentiment", "Unknown")
        session_data.setdefault("description", "No description available")
        session_data.setdefault("timestamp", "Unknown date")
        session_data.setdefault("chat_history", [])
        
        # Convert ObjectId to string
        session_data["_id"] = str(session_data["_id"])
        
        return render_template('session_details.html', 
                            session=session_data,
                            chat_history=session_data.get("chat_history", []))
    except Exception as e:
        print(f"Error loading session details: {e}")
        return "Error loading session details", 500

def get_description(overall_sentiment, overall_score):
    descriptions = {
        "happy": "Keep smiling! Happiness looks great on you.",
        "neutral": "You seem calm and balanced. Stay mindful.",
        "sad": "It's okay to feel down sometimes. Take care of yourself.",
        "angry": "Take a deep breath. Try to relax and clear your mind.",
        "fear": "Fear is natural. Stay calm and face your worries step by step.",
        "disgust": "Not everything is pleasant, but stay positive!",
        "surprise": "Exciting moments can be thrilling! Enjoy the surprises in life."
    }

    description = descriptions.get(overall_sentiment.lower(), "Stay mindful and positive.")

    if overall_sentiment.lower() in ["sad", "fear", "angry", "disgust"] and overall_score < 0.4:
        description += " If you're feeling overwhelmed, consider talking to someone or seeking professional help."

    return description

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

if __name__ == '__main__':
    app.run(debug=True)