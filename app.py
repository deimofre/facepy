ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import dlib
import numpy as np
from skimage import io
from scipy.spatial import distance

# Load the models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


def calculate_similarity(image1_path, image2_path):
    # Load the images
    img1 = io.imread(image1_path)
    img2 = io.imread(image2_path)

    # Detect faces in the images
    dets1 = detector(img1, 1)
    dets2 = detector(img2, 1)

    # Get the shape of the faces
    shape1 = predictor(img1, dets1[0])
    shape2 = predictor(img2, dets2[0])

    # Get the face descriptors
    face_descriptor1 = face_rec_model.compute_face_descriptor(img1, shape1)
    face_descriptor2 = face_rec_model.compute_face_descriptor(img2, shape2)

    # Calculate the Euclidean distance between the two face descriptors
    return distance.euclidean(face_descriptor1, face_descriptor2)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_percentage(similarity):
    percentage = (1 - similarity) * 100
    return percentage
         

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def upload_files():
    image1 = request.files['image1']
    image2 = request.files['image2']

    if image1 and image2 and allowed_file(image1.filename) and allowed_file(image2.filename):
        filename1 = secure_filename(image1.filename)
        filename2 = secure_filename(image2.filename)

        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        image1.save(image1_path)
        image2.save(image2_path)

        similarity = calculate_similarity(image1_path, image2_path)
        similarity_percentage = convert_to_percentage(similarity)

        similarity_percentage = int(similarity_percentage)

        # HTMLテンプレートをレンダリングし、画像のパスと類似度を渡す
        return render_template('result.html', 
                           image1_path=os.path.join('uploads', filename1), 
                           image2_path=os.path.join('uploads', filename2), 
                           similarity_percentage=similarity_percentage)

    return "Invalid file type. Please upload a .png, .jpg, or .jpeg file."


if __name__ == '__main__':
    app.run(debug=True)
