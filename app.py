from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
from tensorflow import keras
from collections import deque
import pyttsx3
import threading
import time

app = Flask(__name__)

# Load both models
models = {
    "alphabet": keras.models.load_model("models/model12.h5"),
    "words": keras.models.load_model("models/wordmodel1.h5")
}

current_model = "alphabet"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define classes
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)
word_list = ['afraid', 'agree', 'assistance', 'bad', 'become', 'college', 'doctor', 'from', 
             'pain', 'pray', 'secondary', 'skin', 'small', 'specific', 'stand', 'today', 
             'warn', 'which', 'work', 'you']

prediction_buffer = deque(maxlen=20)  # Stores letter predictions
word_buffer = []  # Stores confirmed letters
word_prediction_buffer = deque(maxlen=5)  # Stores past word predictions
speech_thread = None

recording_start_time = time.time()
last_hand_detected_time = time.time()
last_spoken_word = ""  # Prevent repeating the same word continuously

CONFIDENCE_THRESHOLD = 0.8  # Higher threshold for accuracy
COLLECTION_TIME = 2.0  # Time to collect the most frequent letter
NO_HAND_TIMEOUT = 2.0  # Time to wait before spelling the word

def speak_text(text):
    """Speak the given text using text-to-speech"""
    def run_tts():
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 170)  # Faster speech
        local_engine.say(text)
        local_engine.runAndWait()
    
    global speech_thread
    if speech_thread and speech_thread.is_alive():
        return
    speech_thread = threading.Thread(target=run_tts)
    speech_thread.start()

def calc_landmark_list(image, landmarks):
    """Extract and normalize hand landmarks"""
    image_width, image_height = image.shape[1], image.shape[0]
    return [[
        min(int(l.x * image_width), image_width - 1),
        min(int(l.y * image_height), image_height - 1),
        l.z
    ] for l in landmarks.landmark]

def pre_process_landmark(landmark_list):
    """Normalize landmark positions for model input"""
    base_x, base_y, base_z = landmark_list[0]
    temp_landmark_list = [[x - base_x, y - base_y, z - base_z] for x, y, z in landmark_list]
    temp_landmark_list = np.array(temp_landmark_list).flatten()
    max_value = max(map(abs, temp_landmark_list)) if temp_landmark_list.any() else 1
    return temp_landmark_list / max_value

def generate_frames():
    """Capture video frames, detect hand gestures, and process sign language"""
    global recording_start_time, last_hand_detected_time, word_buffer, last_spoken_word
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:
        
        while True:
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.flip(image, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            detected_text = ""
            current_time = time.time()
            
            if results.multi_hand_landmarks:
                last_hand_detected_time = current_time  # Reset no-hand timer
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(image, hand_landmarks)
                    processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    df = pd.DataFrame([processed_landmark_list])
                    model = models[current_model]
                    predictions = model(df, training=False).numpy()
                    
                    confidence = np.max(predictions)
                    predicted_class = np.argmax(predictions)
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        if current_model == "alphabet":
                            prediction_buffer.append(alphabet[predicted_class])
                        elif current_model == "words":
                            detected_text = word_list[predicted_class]
                            word_prediction_buffer.append(detected_text)
                            most_common_word = max(set(word_prediction_buffer), key=word_prediction_buffer.count)
                            if most_common_word and most_common_word != last_spoken_word:
                                speak_text(most_common_word)
                                last_spoken_word = most_common_word

                    # Alphabet mode: Store most frequent letter every COLLECTION_TIME seconds
                    if current_model == "alphabet" and current_time - recording_start_time >= COLLECTION_TIME:
                        if prediction_buffer:
                            final_letter = max(set(prediction_buffer), key=prediction_buffer.count)
                            if not word_buffer or word_buffer[-1] != final_letter:  # Prevent duplicate consecutive letters
                                word_buffer.append(final_letter)
                        prediction_buffer.clear()
                        recording_start_time = current_time  # Reset timer
                    
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Alphabet mode: Spell word if no hand detected for NO_HAND_TIMEOUT
            if current_model == "alphabet" and current_time - last_hand_detected_time > NO_HAND_TIMEOUT and word_buffer:
                word = "".join(word_buffer)
                speak_text(word)
                word_buffer.clear()
                prediction_buffer.clear()
                time.sleep(0.1)

            # Display the detected text on the video feed
            text_to_display = "".join(word_buffer) if current_model == "alphabet" else detected_text
            if text_to_display:
                cv2.putText(image, text_to_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html', model=current_model)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global current_model
    current_model = request.form.get("model", "alphabet")
    return "", 204

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
