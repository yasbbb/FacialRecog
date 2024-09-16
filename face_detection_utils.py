import cv2
import numpy as np
from PIL import Image
import os

class VideoCamera(object):
    def __init__(self):
        # Initialize video capture from webcam
        self.video = cv2.VideoCapture(0)
        # Initialize the LBPH face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Load the face detection model (Haarcascade)
        self.cascade_path = "haarcascade_frontalface_default.xml"
        # Path to save/load trained face data
        self.trained_loc = "trainer/trainer.yml"
        # Load the trained model if it exists
        if os.path.exists(self.trained_loc):
            self.recognizer.read(self.trained_loc)
        # Initialize the face detector
        self.faceCascade = cv2.CascadeClassifier(self.cascade_path)

    def __del__(self):
        # Release the video capture when the object is deleted
        self.video.release()

    def create_dataset(self, face_id):
        # Set video frame dimensions
        self.video.set(3, 640)
        self.video.set(4, 480)

        print("\n[INFO] Initializing face capture. Look at the camera...")

        count = 0  # Initialize image count

        while True:
            # Capture frame-by-frame
            ret, img = self.video.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Draw a rectangle around detected face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                # Save the captured face into the dataset folder
                cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])

            # Break if 'ESC' key is pressed or 30 images are captured
            k = cv2.waitKey(100) & 0xff
            if k == 27 or count >= 30:
                break

        print("\n[INFO] Dataset created. Exiting capture mode.")
        self.video.release()
        cv2.destroyAllWindows()

    def getImagesAndLabels(self, path):
        # Function to extract images and their labels from the dataset
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]     
        faceSamples = []  # List to hold face data
        ids = []  # List to hold user IDs
        detector = cv2.CascadeClassifier(self.cascade_path)  # Initialize face detector

        for imagePath in imagePaths:
            # Convert image to grayscale and retrieve ID from filename
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                # Append the face data and associated ID to the lists
                faceSamples.append(img_numpy[y+y+h, x+x+w])
                ids.append(id)

        return faceSamples, ids

    def train_model(self, path):
        # Train the face recognizer with images from the dataset
        print("\n[INFO] Training faces. Please wait...")
        faces, ids = self.getImagesAndLabels(path)  # Extract images and labels
        self.recognizer.train(faces, np.array(ids))  # Train the LBPH recognizer

        # Save the trained model to the trainer directory
        self.recognizer.write('trainer/trainer.yml')
        print(f"\n[INFO] {len(np.unique(ids))} faces trained. Model saved.")

        # Release video and destroy all windows
        self.video.release()
        cv2.destroyAllWindows()

    def run_model(self, names, n_of_persons):
        # Parameters:
        # 'names' --> list of user names to be recognized
        # 'n_of_persons' --> number of users to be recognized (e.g., 1 or 2)

        font = cv2.FONT_HERSHEY_SIMPLEX  # Font for displaying names on screen

        self.video.set(3, 640)  # Set video width
        self.video.set(4, 480)  # Set video height

        # Minimum window size to detect a face
        minW = 0.1 * self.video.get(3)
        minH = 0.1 * self.video.get(4)

        while True:
            # Read a frame from the video feed
            ret, img = self.video.read()

            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                # If an error occurs, restart video capture
                self.video.release()
                cv2.destroyAllWindows()
                self.video = cv2.VideoCapture(0)
                ret, img = self.video.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                # Draw rectangle around detected face
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Predict the face using the recognizer
                id, confidence = self.recognizer.predict(gray[y+y+h, x+x+w])

                # Calculate confidence score
                confidence_calc = 100 - confidence
                if confidence_calc > 0:
                    id = names[id]  # Retrieve user name
                    confidence = f"  {round(confidence_calc)}%"
                else:
                    id = "unknown"
                    confidence = f"  {round(confidence_calc)}%"

                # Display the name and confidence on the screen
                cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

            # Encode the frame and return as a byte stream
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
