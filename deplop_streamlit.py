import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load model và khởi tạo mediapipe
mp_hands = mp.solutions.hands
model = pickle.load(open('MLPClassifier/MLP_model.p', 'rb'))

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x)
                    data_aux.append(lm.y)
                    x_.append(lm.x)
                    y_.append(lm.y)
                x1 = int(min(x_) * img.shape[1])
                y1 = int(min(y_) * img.shape[0])
                prediction = model.predict([np.array(data_aux)])
                cv2.putText(img, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Realtime Sign Language Recognition")
webrtc_streamer(key="sign-language", video_processor_factory=SignLanguageProcessor)