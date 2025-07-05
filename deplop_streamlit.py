import cv2
import pickle
import numpy as np
import streamlit as st
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)
model = pickle.load(open('MLPClassifier/MLP_model.p', 'rb'))

num_classes = len(model.classes_)

st.title("Real-time SIgn Language Recognition")
frame_window=st.image([])

run=st.checkbox('Start Camera')

camera = cv2.VideoCapture(0)

while run:
    x_ = []
    y_ = []
    ret, frame = camera.read()
    if not ret:
        break
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)
                x_.append(lm.x)
                y_.append(lm.y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        prediction = model.predict([np.array(data_aux)])
        probs = model.predict_proba([np.asarray(data_aux)])[0]
        # current_class = prediction[0]

        st.markdown(f'**Prediction:** {prediction[0]}')
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

camera.release()
cv2.destroyAllWindows()