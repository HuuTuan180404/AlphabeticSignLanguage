const videoElement = document.getElementById("video");
const resultText = document.getElementById("letter");
const confidenceDetection = document.getElementById("confidence");
const inputText = document.getElementById("resultConsole");

let lastPredicted = null;
let stableSince = null;
const STABLE_TIME = 2000;

// Khởi tạo MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
});
hands.onResults(onResults);

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  videoElement.srcObject = stream;

  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480,
  });
  camera.start();
}

function onResults(results) {
  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0)
    return;

  const landmarks = results.multiHandLandmarks[0];
  const flatData = [];
  landmarks.forEach((point) => {
    flatData.push(point.x);
    flatData.push(point.y);
  });

  sendLandmarksToServer(flatData);
}

async function sendLandmarksToServer(landmarks) {
  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ landmarks }),
    });

    const result = await response.json();
    resultText.innerText = `${result.class}`;
    confidenceDetection.innerText = ` ${(result.probs[result.class] * 100).toFixed(2)}%`;

    const predictedClass = result.class;

    const now = Date.now();

    if (predictedClass === lastPredicted) {
            if (now - stableSince >= STABLE_TIME) {
                // Đủ 2s ổn định → xử lý
                if (predictedClass === "space") {
                    inputText.value += " ";
                } else if (predictedClass === "del") {
                    inputText.value = inputText.value.slice(0, -1);
                } else {
                    inputText.value += predictedClass;
                }
                // Reset để tránh ghi lặp lại liên tục
                lastPredicted = null;
                stableSince = null;
            }
        } else {
            // ký hiệu mới → bắt đầu đếm thời gian
            lastPredicted = predictedClass;
            stableSince = now;
        }

  } catch (err) {
    console.error("Error sending data to server:", err);
  }
}
startCamera();
