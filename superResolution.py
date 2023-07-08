import sys
import keras
import cv2
import numpy
import matplotlib
import skimage
import math
import sys
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.optimizers import SGD, Adam
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib
import numpy as np
import os
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

print('Python: {}'.format(sys.version))
print('Keras: {}'.format(keras.__version__))
print('OpenCV: {}'.format(cv2.__version__))
print('Numpy: {}'.format(numpy.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('skimage: {}'.format(skimage.__version__))
print('Python: {}'.format(sys.version))
print('Keras: {}'.format(keras.__version__))
print('OpenCV: {}'.format(cv2.__version__))
print('Numpy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))

#define a function for peak signal to noise ratio

def psnr(target, ref):
    # assume an RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    if rmse == 0:
        return float('inf')
    else:
        return 20 * math.log10(255. / rmse)


#define function for mean square error

def mse(target, ref):
    # mse is the sum of squared differences divided by the number of pixels
    err = np.mean((target.astype(float) - ref.astype(float)) ** 2)
    return err

#define function that combines all three image qualit metrics

from skimage.metrics import structural_similarity as compare_ssim

def compare_images(target, ref):
    scores = []

    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))

    target_height, target_width = target.shape[:2]
    max_win_size = min(target_height, target_width)

    # Adjust the fraction to your preference, but make sure win_size is odd
    win_size = 15 // 8
    win_size = max(win_size, 3)  # Set a minimum value of 3

    # Make sure win_size is odd
    if win_size % 2 == 0:
        win_size += 1

    scores.append(compare_ssim(target, ref, win_size=win_size, multichannel=True))

    return scores


#function to dewgrade image
def prepare_images(path, factor):
    # Loop through the files in the directory
    for file in os.listdir(path):
        # Open the file
        img = cv2.imread(os.path.join(path, file))

        # Find old and new image dimensions
        h, w, c = img.shape
        new_height = int(h / factor)
        new_width = int(w / factor)

        # Resize the image - down
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Resize the image - up
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # Save the image
        print('Saving {}'.format(file))
        cv2.imwrite('image/{}'.format(file), img)



prepare_images('G:\desktop\DEEP_LEARNING\source', 2)


for file in os.listdir('image/'):
    # Open target and reference images
    target = cv2.imread('image/{}'.format(file))
    ref = cv2.imread('source/{}'.format(file))

    # Calculate the scores
    scores = compare_images(target, ref)

    # Print all three scores
    print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))


from keras.layers import Activation
# define our SRCN model

def model():
  #define model type
  SRCNN=Sequential()

  #add model layer
  SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='glorot_uniform', activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
  SRCNN.add(Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='glorot_uniform',
                   activation='relu',padding='same',use_bias=True))
  SRCNN.add(Conv2D(filters=1,kernel_size=(5,5),kernel_initializer='glorot_uniform',activation='linear',padding='valid',use_bias=True))


  #define optimizer
  adam=Adam(lr=0.0003)

  #compile the model
  SRCNN.compile(optimizer=adam,loss='mean_squared_error',metrics=['mean_squared_error'])
  return SRCNN


#define necccesary image processing functions

def modcrop(img, scale):
    sz = img.shape[:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 0:sz[1], :]
    return img



def shave(image,border):
  img=image[border: -border,border:-border]
  return img




def predict(image_path, model):
    # Rest of the function code remains the same
    # Load the srcnn model with weights
    srcnn = model
    # Rest of the code
    srcnn.load_weights('3051crop_weight_200.h5')

    # Load the low resolution image
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    ref = cv2.imread('source/{}'.format(file))

    # PREPROCESSING THE IMAGE
    ref = modcrop(ref, 3)
    degraded = modcrop(degraded, 3)

    # Convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255

    # Perform super-resolution with srcnn
    pre = srcnn.predict(Y, batch_size=1)

    # Post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)

    # Copy Y channel back to image and convert to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

    # Remove border from reference and degraded image
    ref = shave(ref.astype(np.uint8), 6)
    degraded = shave(degraded.astype(np.uint8), 6)

    # Image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref))
    scores.append(compare_images(output, ref))

    # Return images and scores
    return ref, degraded, output, scores


# Load or create the SRCNN model
srcnn = model()

# Load the weights into the model
srcnn.load_weights('3051crop_weight_200.h5')

# Save the model
srcnn.save('model.h5')

#PASSING tHE IMAGE to model



import tkinter as tk
from tkinter import filedialog
import os
import shutil
from keras.models import load_model

# Load the saved model
model = load_model('model.h5')

# Create a Tkinter window
window = tk.Tk()
window.withdraw()  # Hide the main window

# Ask the user to select an image file
image_path = filedialog.askopenfilename(title='Select an image file', filetypes=[('Image files', '*.jpg;*.jpeg;*.png;*.bmp')])

# Check if an image was selected
if image_path:
    # Save the selected image into the 'image' folder
    image_filename = os.path.basename(image_path)
    image_save_path = os.path.join('image', image_filename)
    shutil.copyfile(image_path, image_save_path)

    # Save the selected image into the 'source' folder
    source_save_path = os.path.join('source', image_filename)
    shutil.copyfile(image_path, source_save_path)

    # Call the predict function with the modified image path and loaded model
    ref, degraded, output, scores = predict(image_save_path, model)

    # Delete the uploaded image file from source folder
    os.remove(source_save_path)

    # Delete the processed image file from image folder
    os.remove(image_save_path)

    # Display images as subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Degraded')
    axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')

    # Remove the x and y tick marks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()













