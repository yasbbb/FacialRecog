from flask import Flask, render_template, Response, request
from face_detection_utils import VideoCamera

# Initialize Flask app
app = Flask(__name__)

# Define a global list to hold the names of the users
names = []

@app.route('/', methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            req = request.form
            User_1 = req.get("User1")  # Get User1 name from form
            global names
            names = ["", User_1]  # Store User1 name globally

            if 'Run' in request.form:
                return render_template('index.html')

            if 'Create_Dataset' in request.form:
                # Add logging to catch errors here
                print(f"Creating dataset for User 1: {User_1}")
                face_id = 1
                camera = VideoCamera()
                camera.create_dataset(face_id)
                return render_template('index.html')

            if 'Train_Dataset' in request.form:
                camera = VideoCamera()
                camera.train_model("dataset")
                return render_template('processing.html')

    except Exception as e:
        # Log the exception to get more details
        print(f"Error during dataset creation for User 1: {e}")
        error = "An error occurred. Please read the instructions and repeat the process."
        return render_template('ErrorPage.html')



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
