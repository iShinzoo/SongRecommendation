import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os
import logging
import webbrowser

# Set up logging
logging.basicConfig(filename="app_log.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Log app start
logging.info("Application started.")

# Load the emotion model and labels
model_path = "model.h5"
labels_path = "labels.npy"

try:
    if os.path.exists(model_path) and os.path.exists(labels_path):
        model = load_model(model_path, compile=False)  # Suppress compile warning
        labels = np.load(labels_path)
        logging.info("Model and labels loaded successfully.")
    else:
        raise FileNotFoundError("Model or labels file not found.")
except Exception as e:
    logging.error(f"Error loading model or labels: {e}")
    st.error(f"Error loading model or labels: {e}")
    st.stop()

# Mediapipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Streamlit header
st.header("Emotion-Based Music Recommender")

# Emotion Processor class
class EmotionProcessor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic()

    def process_frame(self, frm):
        try:
            frm = cv2.flip(frm, 1)  # Mirror the frame for a better user experience

            # Process the frame and extract landmarks
            results = self.holistic.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            landmarks = []

            # Extract facial landmarks
            if results.face_landmarks:
                for lm in results.face_landmarks.landmark:
                    landmarks.append(lm.x - results.face_landmarks.landmark[1].x)
                    landmarks.append(lm.y - results.face_landmarks.landmark[1].y)

            # Left Hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks.append(lm.x - results.left_hand_landmarks.landmark[8].x)
                    landmarks.append(lm.y - results.left_hand_landmarks.landmark[8].y)
            else:
                # Pad with zeros for missing left hand landmarks
                landmarks.extend([0.0] * 42)

            # Right Hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks.append(lm.x - results.right_hand_landmarks.landmark[8].x)
                    landmarks.append(lm.y - results.right_hand_landmarks.landmark[8].y)
            else:
                # Pad with zeros for missing right hand landmarks
                landmarks.extend([0.0] * 42)

            # Ensure the expected shape for the model (1020 features)
            while len(landmarks) < 1020:
                landmarks.append(0.0)  # Padding with zeros if fewer features

            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict emotion
            pred = labels[np.argmax(model.predict(landmarks))]
            
            # Store the predicted emotion in session state
            st.session_state.predicted_emotion = pred

            # Add emotion text on top of the frame using OpenCV
            cv2.putText(frm, f"Emotion: {pred}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Return the frame with predicted emotion
            return frm
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frm

# User inputs for song recommendation
lang = st.text_input("Preferred Language (e.g., English, Hindi):")
singer = st.text_input("Favorite Singer:")

# Show WebRTC or Webcam capture after user inputs language and singer
if lang and singer:
    st.session_state["run"] = True  # Allow starting the Webcam when user inputs are provided

    try:
        # Webcam capture instead of WebRTC
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()  # This is where the webcam video will be displayed

        # Create a container for the video feed to keep it separate from the button
        video_container = st.container()

        # Place the button in the container at the top of the screen
        with st.container():  # Always place the button container here
            if st.button("Recommend Me Songs"):
                try:
                    # Get the predicted emotion from session state
                    predicted_emotion = st.session_state.get("predicted_emotion", None)

                    if not predicted_emotion:
                        st.warning("Emotion not detected. Please try again.")
                    elif not lang or not singer:
                        st.warning("Please fill in the language and singer details.")
                    else:
                        query = f"{lang} {predicted_emotion} songs {singer}"
                        st.success(f"Searching for: {query}")
                        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
                        np.save("emotion.npy", np.array([""]))  # Clear the stored emotion after search
                        st.session_state["run"] = False  # Stop further emotion capturing
                        logging.info(f"User requested songs based on: {query}")
                except Exception as e:
                    logging.error(f"Error recommending songs: {e}")
                    st.error(f"Error recommending songs: {e}")
        
        # Display the webcam feed in the video container
        with video_container:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error capturing frame from webcam.")
                    break

                # Process the frame using the emotion processor
                processor = EmotionProcessor()
                frame = processor.process_frame(frame)

                # Show the frame with emotion prediction (on top of video feed)
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                # Exit loop if user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Error capturing webcam: {e}")
        st.error(f"Error capturing webcam: {e}")
