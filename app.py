from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_from_directory
import mysql.connector
from mysql.connector import errorcode
import os
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a random secret key for session management

# Database configuration
db_config = {
    'user': 'root',
    'password': 'Chisum@7',
    'host': 'localhost',
    'database': 'helmet_detection',
}


# Connect to the database
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)  # Establish a connection to the database
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")  # Handle access denied error
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")  # Handle database not found error
        else:
            print(err)  # Print other errors


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Set the upload folder for storing uploaded files

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create the upload folder if it doesn't exist

# YOLO model paths (update these paths as needed)
weights0_path = r'D:\PyCharm\projects\Helmet-Detection\Helmet detection\helmet_detection\datasets\detect-person-on-motorbike-or-scooter\yolov3-obj_final.weights'
configuration0_path = r'D:\PyCharm\projects\Helmet-Detection\Helmet detection\helmet_detection\datasets\detect-person-on-motorbike-or-scooter\yolov3_pb.cfg'
weights1_path = r'D:\PyCharm\projects\Helmet-Detection\Helmet detection\helmet_detection\datasets\helmet-detection-yolov3\yolov3-helmet.weights'
configuration1_path = r'D:\PyCharm\projects\Helmet-Detection\Helmet detection\helmet_detection\datasets\helmet-detection-yolov3\yolov3-helmet.cfg'
labels0_path = r'D:\PyCharm\projects\Helmet-Detection\Helmet detection\helmet_detection\datasets\detect-person-on-motorbike-or-scooter\coco.names'
labels1_path = r'D:\PyCharm\projects\Helmet-Detection\Helmet detection\helmet_detection\datasets\helmet-detection-yolov3\helmet.names'

# Load the YOLO models and labels
network0 = cv2.dnn.readNetFromDarknet(configuration0_path, weights0_path)
network1 = cv2.dnn.readNetFromDarknet(configuration1_path, weights1_path)
labels0 = open(labels0_path).read().strip().split('\n')
labels1 = open(labels1_path).read().strip().split('\n')
layers_names0_output = [network0.getLayerNames()[i - 1] for i in network0.getUnconnectedOutLayers()]
layers_names1_output = [network1.getLayerNames()[i - 1] for i in network1.getUnconnectedOutLayers()]

probability_minimum = 0.5  # Minimum probability to eliminate weak detections
threshold = 0.3  # Threshold for non-maximum suppression


@app.route('/')
def index():
    if 'user_id' not in session:  # Check if the user is not logged in
        return redirect(url_for('login'))  # Redirect to login page if not logged in
    return render_template('index.html')  # Render the index page


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user and check_password_hash(user['password'], password):  # Verify password
            session['user_id'] = user['id']  # Set session variable for user ID
            return redirect(url_for('index'))  # Redirect to index page after successful login
        else:
            return render_template('login_error.html')  # Show error page if login fails
    return render_template('login.html')  # Render the login page


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Hash the password using the default method (pbkdf2:sha256)
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)', (name, email, hashed_password))
        conn.commit()  # Commit the transaction
        cursor.close()
        conn.close()

        return redirect(url_for('login'))  # Redirect to login page after successful signup

    return render_template('signup.html')  # Render the signup page


@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user_id from session to log out
    return redirect(url_for('login'))  # Redirect to login page


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Insert feedback into the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO feedback (name, email, message) VALUES (%s, %s, %s)', (name, email, message))
        conn.commit()  # Commit the transaction
        cursor.close()
        conn.close()

        return redirect(url_for('contact'))  # Redirect to contact page after submitting feedback

    return render_template('contact.html')  # Render the contact page


@app.route('/about')
def about():
    if 'user_id' not in session:  # Check if the user is not logged in
        return redirect(url_for('login'))  # Redirect to login page if not logged in
    return render_template('about.html')  # Render the about page


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:  # Check if the user is not logged in
        return redirect(url_for('login'))  # Redirect to login page if not logged in

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})  # Return error if no file part in the request

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})  # Return error if no file is selected

    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save the uploaded file

        image_input = cv2.imread(file_path)
        h, w = image_input.shape[:2]
        blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        network0.setInput(blob)
        network1.setInput(blob)

        output_from_network0 = network0.forward(layers_names0_output)
        output_from_network1 = network1.forward(layers_names1_output)

        np.random.seed(42)
        colours0 = np.random.randint(0, 255, size=(len(labels0), 3), dtype='uint8')
        colours1 = np.random.randint(0, 255, size=(len(labels1), 3), dtype='uint8')

        bounding_boxes0 = []
        confidences0 = []
        class_numbers0 = []

        bounding_boxes1 = []
        confidences1 = []
        class_numbers1 = []

        # Process detection results for person detection
        for result in output_from_network0:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    box_current = detection[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes0.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences0.append(float(confidence_current))
                    class_numbers0.append(class_current)

        # Process detection results for helmet detection
        for result in output_from_network1:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    box_current = detection[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes1.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences1.append(float(confidence_current))
                    class_numbers1.append(class_current)

        # Non-Maximum Suppression for person detection
        results0 = cv2.dnn.NMSBoxes(bounding_boxes0, confidences0, probability_minimum, threshold)
        if len(results0) > 0:
            for i in results0.flatten():
                x_min, y_min = bounding_boxes0[i][0], bounding_boxes0[i][1]
                box_width, box_height = bounding_boxes0[i][2], bounding_boxes0[i][3]
                colour_box_current = [int(j) for j in colours0[class_numbers0[i]]]
                cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current,
                              5)
                text_box_current0 = '{}: {:.4f}'.format(labels0[int(class_numbers0[i])], confidences0[i])
                cv2.putText(image_input, text_box_current0, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            colour_box_current, 5)

        # Non-Maximum Suppression for helmet detection
        results1 = cv2.dnn.NMSBoxes(bounding_boxes1, confidences1, probability_minimum, threshold)
        if len(results1) > 0:
            for i in results1.flatten():
                x_min, y_min = bounding_boxes1[i][0], bounding_boxes1[i][1]
                box_width, box_height = bounding_boxes1[i][2], bounding_boxes1[i][3]
                colour_box_current = [int(j) for j in colours1[class_numbers1[i]]]
                cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current,
                              5)
                text_box_current1 = '{}: {:.4f}'.format(labels1[int(class_numbers1[i])], confidences1[i])
                cv2.putText(image_input, text_box_current1, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            colour_box_current, 5)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + file.filename)
        cv2.imwrite(output_path, image_input)  # Save the image with detections

        return jsonify({'message': 'Detection complete',
                        'image_url': 'uploads/output_' + file.filename})  # Return the URL of the processed image
    else:
        return jsonify({'error': 'Invalid file type'})  # Return error if the file type is invalid


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)  # Serve the uploaded file


if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application in debug mode
# you can use host="192.168.137.1", port=8000
