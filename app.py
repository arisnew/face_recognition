from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import threading
import base64

app = Flask(__name__)

# Global variables for training
is_training = False
training_thread = None

# Webcam
camera = cv2.VideoCapture(0)  # 0 represents the default camera
image_count = 0
# Directory to save captured images
capture_dir = 'captured_images'
# Create the directory if it doesn't exist
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)
# Create a directory for storing captured images
if not os.path.exists('dataset'):
    os.mkdir('dataset')
# Directory to save the trained model
trained_model_path = 'trained_model.xml'
# Load the trained model and other configurations
trained_model = cv2.face_LBPHFaceRecognizer.create()
trained_model.read('trained_model.xml')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('train.html')

@app.route('/selfie')
def selfie():
    return render_template('capture.html')

@app.route('/recognize', methods=['POST'])
def recognize_face():
    data = request.get_json()
    # image_data = data['image']  # Assuming that 'image' is a base64-encoded image string in the request

    # print(data)

    # Decode the base64-encoded image data to binary
    # binary_data = base64.b64decode(image_data)

    # print(binary_data)

    # versi opencv
    global camera
    ret, frame = camera.read()
    # img = frame # cv2.imdecode(frame, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img = cv2.LBP(gray_image)

    # Convert binary data to a NumPy array
    # image_array = np.frombuffer(binary_data, np.uint8)
    # Decode the image using OpenCV
    # img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # img = cv2.imdecode(np.frombuffer(bytes(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.IMREAD_COLOR)


    # Perform face recognition (you need to implement this part)
    label, confidence = trained_model.predict(img)

    return jsonify({'result': f'Label: {label}, Confidence: {confidence}'})


@app.route('/capture', methods=['POST'])
def capture_face():
    global camera
    ret, frame = camera.read()

    if not ret:
        return jsonify({'message': 'Failed to capture image'})

    # Prompt the user for the person's identity
    data = request.json
    person_identity = data.get('identity')  # Assumes a form field named 'identity'

    print(person_identity)
    if not person_identity:
        return jsonify({'message': 'Person identity not provided'})

    # Create a subdirectory for the person if it doesn't exist
    person_dir = os.path.join(capture_dir, person_identity)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Generate a unique filename for the captured image
    image_count = len(os.listdir(person_dir))
    image_path = os.path.join(person_dir, f'{person_identity}_image{image_count}.jpg')

    # Save the frame as an image
    cv2.imwrite(image_path, frame)

    return jsonify({'message': 'Image captured successfully'})
    
@app.route('/train', methods=['POST'])
def train_face_recognition():
    recognizer = cv2.face_LBPHFaceRecognizer.create()

    face_images = []
    labels = []

    for subdir, dirs, files in os.walk(capture_dir):
        for dir_name in dirs:
            label = int(dir_name)
            for filename in os.listdir(os.path.join(capture_dir, dir_name)):
                img_path = os.path.join(capture_dir, dir_name, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                face_images.append(img)
                labels.append(label)

    # Train the face recognition model
    recognizer.train(face_images, np.array(labels))

    # Save the trained model
    recognizer.save(trained_model_path)

    return jsonify({'message': 'Training completed successfully'})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
