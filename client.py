from flask import Flask, render_template, Response, request
from face_detection_utils import VideoCamera

# Initialize Flask app
app = Flask(__name__)

# Define a global list to hold the names of the users
names = []

@app.route('/', methods=["GET", "POST"])
def index():
    # Handle both GET and POST requests
    if request.method == "POST":
        req = request.form
        User_1 = req.get("User1")  # Get User1 name from form
        global names
        names = ["", User_1]  # Store User1 name globally

        if 'Run' in request.form:
            # Render the main template to start the video feed
            return render_template('index.html')

        if 'Create_Dataset' in request.form:
            # Create a dataset for the first user
            face_id = 1
            camera = VideoCamera()  # Initialize camera
            camera.create_dataset(face_id)  # Capture 30 face images for training
            return render_template('index.html')

        if 'Train_Dataset' in request.form:
            # Train the face recognition model using the dataset
            camera = VideoCamera()  # Initialize camera
            camera.train_model("dataset")  # Train the model using images
            return render_template('processing.html')

    # Default render for GET requests
    return render_template('index.html')


# Function to generate video feed
def gen(camera):
    while True:
        # Run the model and capture frame
        frame = camera.run_model(names, 1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed', methods=["GET", "POST"])
def video_feed():
    # Route for video streaming
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Run the Flask app locally
    app.run(host='127.0.0.1', debug=True)
