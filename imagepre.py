import cv2
import numpy as np
import pandas as pd

# Function to preprocess image
def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize image to a fixed size (100x100)
    resized_image = cv2.resize(image, (100, 100))
    
    # Normalize pixel values
    normalized_image = resized_image / 255.0

    # Calculate normalized entropy (sample feature)
    histogram = cv2.calcHist([resized_image], [0], None, [256], [0, 256])
    histogram = histogram / np.sum(histogram)
    normalized_entropy = -np.sum(histogram * np.log2(histogram + 1e-7))

    # Example feature vector (mean pixel value and entropy)
    image_features = [np.mean(normalized_image), normalized_entropy]

    return np.array(image_features)
