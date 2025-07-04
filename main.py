import cv2
import pickle
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)
model = pickle.load(open('MLPClassifier/MLP_model.p', 'rb'))

num_classes = len(model.classes_)

cap = cv2.VideoCapture(0)

while True:
    x_ = []
    y_ = []
    ret, frame = cap.read()
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
        current_class = prediction[0]

        cv2.putText(frame, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    panel_height = max(H, num_classes * 30)
    output_panel = np.ones((panel_height, 250, 3), dtype=np.uint8) * 255

    if results.multi_hand_landmarks:
        label_probs = list(zip(model.classes_, probs))
        label_probs.sort(key=lambda x: x[1], reverse=True)

        for idx, (label, prob) in enumerate(label_probs):
            y_pos = 50 + idx * 24
            cv2.putText(output_panel, f"{label}: {prob * 100:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    frame_resized = cv2.resize(frame, (W, panel_height))
    combined = np.hstack((frame_resized, output_panel))
    cv2.imshow('Realtime Hand Detection', combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()