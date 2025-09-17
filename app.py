# backend/app.py

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import io
import pyttsx3
from deep_translator import GoogleTranslator
import mediapipe as mp # Import MediaPipe

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the NEW, more accurate model
model = load_model('../ml_model/trained_model/asl_model_mobilenet.h5')

# Create a mapping from index to class name
class_names = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                'Y', 'Z', 'del', 'nothing', 'space']

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_image(image_data):
    """Decodes image, finds the hand using MediaPipe, crops it, and prepares it for the model."""
    try:
        img_data = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Process the frame with MediaPipe to find hands
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return None  # No hand detected

        # Get bounding box coordinates for the detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, c = frame.shape
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

        # Add some padding to the bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop the image to just the hand
        cropped_hand = frame[y_min:y_max, x_min:x_max]

        if cropped_hand.size == 0:
            return None

        # Resize and normalize the cropped hand image for the model
        resized_frame = cv2.resize(cropped_hand, (64, 64))
        processed_image = np.expand_dims(resized_frame, axis=0) / 255.0
        return processed_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@socketio.on('image')
def handle_image(data):
    """Receives an image, preprocesses it to find the hand, predicts, and sends back the result."""
    image_data = data.split(',')[1]
    processed_image = preprocess_image(image_data)

    if processed_image is not None:
        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

        if confidence > 0.8: # We can use a higher confidence threshold now
            emit('prediction', {'class': predicted_class, 'confidence': confidence})

@app.route('/speak', methods=['POST'])
def speak():
    """Converts text to speech."""
    data = request.get_json()
    text = data.get('text', '')
    lang = data.get('lang', 'en')

    if not text:
        return jsonify({"status": "error", "message": "No text provided"}), 400

    if lang == 'hi':
        try:
            translated_text = GoogleTranslator(source='auto', target='hi').translate(text)
            engine.say(translated_text)
        except Exception as e:
            print(f"Translation/Hindi speech error: {e}")
            engine.say("Could not speak in Hindi.")
    else:
        engine.say(text)

    engine.runAndWait()
    return jsonify({"status": "success"})


if __name__ == '__main__':
    # Make sure to install eventlet for WebSocket support
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)

