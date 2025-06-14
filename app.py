import os
from flask import Flask, render_template, Response
import cv2
from mask_detection_script import detect_and_predict_mask

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_and_predict_mask(frame)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/facedetection')
def facedetection():
    return render_template('face_detection.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health_check():
    return {"status": "healthy", "message": "Face Mask Detection App is running"}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)