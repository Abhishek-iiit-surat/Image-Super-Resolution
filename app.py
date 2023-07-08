import os
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from keras.models import load_model
import cv2
import numpy as np

# Create the Flask application
app = Flask(__name__,static_folder='static')
UPLOAD_FOLDER = 'uploads'
RECONSTRUCTED_FOLDER = 'reconstructed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECONSTRUCTED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECONSTRUCTED_FOLDER'] = RECONSTRUCTED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# Load your model
model = load_model('G:\desktop\DEEP_LEARNING\model.h5')

# Define a function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the file upload route
@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file extension.')

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Load and preprocess the uploaded image
    uploaded_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # preprocessed_image = preprocess_image(uploaded_image)

    # Pass the input image to the model for prediction
    reconstructed_image = predict(uploaded_image, model)

    # Post-process the reconstructed image
    # postprocessed_image = postprocess_image(reconstructed_image)

    # Save the reconstructed image
    reconstructed_filename = 'reconstructed_' + filename
    reconstructed_filepath = os.path.join(app.config['RECONSTRUCTED_FOLDER'], reconstructed_filename)
    cv2.imwrite(reconstructed_filepath, reconstructed_image)

    # Return the reconstructed image for download
    return send_file(reconstructed_filepath, as_attachment=True)

def preprocess_image(image):
    # Resize the image
    image = modcrop(image, 3)
    
    # Convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = np.zeros((1, image.shape[0], image.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = image[:, :, 0].astype(float) / 255

    return Y

def postprocess_image(image):
    # Copy Y channel back to image and convert to BGR
    temp = np.copy(image)
    temp = shave(temp, 6)
    temp[:, :, 0] = image[0, :, :, 0]
    temp = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

    # Remove border from image
    output = shave(temp, 6)

    return output

def modcrop(img, scale):
    sz = img.shape[:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 0:sz[1], :]
    return img

def shave(image, border):
    img = image[border:-border, border:-border]
    return img

def predict(image, model):
    # Preprocess the image (if necessary)
    # ...
    degraded = modcrop(image, 3)
    # Convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    # Perform super-resolution with srcnn
    pre = model.predict(Y,batch_size=1);

    # Post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)

    # Copy Y channel back to image and convert to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    return output

if __name__ == '__main__':
    app.run(debug=True)
