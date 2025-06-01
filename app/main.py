from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open('../MLPClassifier/MLP_model.p', 'rb'))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    landmarks = data.get("landmarks", [])

    if len(landmarks) != 42:  # 21 điểm, mỗi điểm gồm x, y
        return jsonify({"error": "Invalid input"}), 400

    prediction = model.predict([np.array(landmarks)])
    probs = model.predict_proba([np.array(landmarks)])[0]

    return jsonify({
        "class": prediction[0],
        "probs": dict(zip(model.classes_, probs.round(4).tolist()))
    })

if __name__ == "__main__":
    app.run(debug=True)
