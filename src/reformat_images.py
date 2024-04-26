import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def jpg_to_numpy(read_path, write_path):
    target_size = (224, 224)
    output_arr = []
    for filename in os.listdir(read_path):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            file_path = os.path.join(read_path, filename)
            image = Image.open(file_path)
            image = image.resize(target_size)
            image = np.array(image) / 255.0

            if(image.shape == (224, 224, 3)):
                image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
                print(image.shape)
            if(image.shape == (224, 224)):
                output_arr.append(image)
    np.save(write_path, np.array(output_arr))

def load_data():
    jpg_to_numpy("../MRI_Data/Training/notumor/", "../MRI_Data/Training/notumor.npy")
    jpg_to_numpy("../MRI_Data/Training/pituitary/", "../MRI_Data/Training/pituitary.npy")
    jpg_to_numpy("../MRI_Data/Training/meningioma/", "../MRI_Data/Training/meningioma.npy")
    jpg_to_numpy("../MRI_Data/Training/glioma/", "../MRI_Data/Training/glioma.npy")

def test_numpy_files():
    test = np.load("../MRI_Data/Training/pituitary.npy")
    print(test[0].shape)
    plt.imshow(test[11], cmap='gray')
    plt.axis('off')
    plt.show()

def generate_dataset():
    print("building")
#load_data()
#test_numpy_files()

