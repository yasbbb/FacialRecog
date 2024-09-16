from flask import Flask, render_template, Response, request, flash
from face_detection_utils import VideoCamera
import os

app = Flask(__name__)
app.secret_key = 'eRqUle$'

global names, n_of_users
# Initialize the VideoCamera instance globally
Instance = VideoCamera()

@app.route('/', methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            req = request.form
            user_1 = req.get("User1")
            user_2 = req.get("User2")

            if 'Erase_Data' in request.form:
                # Erase all user data from the dataset
                dirs = os.listdir("dataset/")
                for file in dirs:
                    os.remove("dataset/" + file)

            if 'Create_Dataset_User1' in request.form:
                # Create dataset for User 1
                if user_1 == "":
                    error = "You need to type the name of User 1 to create the dataset"
                    flash(error)
                    return render_template('index.html')
                else:
                    face_id = 1
                    Instance.create_dataset(face_id)  # Capture images for User 1
                    Instance.train_model("dataset")  # Train the model with the dataset
                    return render_template('index.html')

            if 'Create_Dataset_User2' in request.form:
                # Create dataset for User 2
                if user_2 == "":
                    error = "You need to type the name of User 2 to create the dataset"
                    flash(error)
                    return render_template('index.html')
                else:
                    face_id = 2
                    Instance_Dataset2 = VideoCamera()
                    Instance_Dataset2.create_dataset(face_id)  # Capture images for User 2
                    Instance_Dataset2.train_model("dataset")  # Train the model with the dataset
                    return render_template('index.html')

            if 'Run' in request.form:
                # Check if both users' datasets are ready and run the model
                if user_1 == "" and user_2 == "":
                    error = "You need to type the names of User 1 and User 2 to execute"
                    flash(error)
                    return render_template('index.html')

                if user_1 == "":
                    error = "You need to type the name of User 1 to execute"
                    flash(error)
                    return render_template('index.html')

                if user_2 == "":
                    user_1_path = "dataset/User.1.2.jpg"
                    if os.path.exists(user_1_path):
                        global names, n_of_users
                        names = ["", user_1]  # Set only User 1 for recognition
                        n_of_users = 1
                        return render_template('index_camera.html')
                    else:
                        error = "You need to create the dataset of User 1 to execute"
                        flash(error)
                        return render_template('index.html')

                else:
                    user_1_path = "dataset/User.1.2.jpg"
                    user_2_path = "dataset/User.2.2.jpg"
                    if os.path.exists(user_1_path) and os.path.exists(user_2_path):
                        names = ["", user_1, user_2]  # Set both User 1 and User 2 for recognition
                        n_of_users = 2
                        return render_template('index_camera.html')
                    else:
                        error = "You need to create the dataset of both users to execute"
                        flash(error)
                        return render_template('index.html')

    except Exception as e:
        error = "An error occurred. Please read the instructions and repeat the process"
        print(e)
        flash(error)
        return render_template('ErrorPage.html')

    return render_template('index.html')

# Function to generate video feed for two users
def gen():
    while True:
        frame = Instance.run_model(names, n_of_users)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed', methods=["GET", "POST"])
def video_feed():
    # Route for video streaming
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)
