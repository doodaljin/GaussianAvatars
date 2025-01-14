import os
from PIL import Image
import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def load_images(image_dir):
    """Loads images from a directory and returns a dictionary of image data.

    Args:
        image_dir: Path to the directory containing images.

    Returns:
        A dictionary where keys are image file paths and values are PIL Image objects.
    """
    images = {}
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                image_path = os.path.join(image_dir, filename)
                img = Image.open(image_path)
                img = img.convert('RGB')  # Ensure images are in RGB format
                images[image_path] = img
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return images



def extract_features(images, feature_type='hog', resize_size=(100, 100)):
   """Extracts image features using either HOG or a simple pixel grid method.

   Args:
       images: Dictionary of image paths and PIL Image objects.
       feature_type: 'hog' for HOG features or 'pixel' for pixel values.
       resize_size: Size to resize images before feature extraction.

   Returns:
       A dictionary where keys are image paths and values are extracted feature vectors.
   """
   features = {}
   for image_path, img in images.items():
       try:
           img_resized = img 
           if feature_type == 'hog':
               img_gray = np.array(img_resized.convert('L')) # Convert to grayscale
               hog_features = hog(img_gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) # Adjust parameters if needed
               features[image_path] = hog_features
           elif feature_type == 'pixel':
               img_array = np.array(img_resized).flatten() # Flatten RGB to a vector of pixel values
               features[image_path] = img_array
           else:
              raise ValueError("Invalid feature_type")
       except Exception as e:
          print(f"Error extracting features from {image_path}: {e}")
   return features

def find_similar_images(features, threshold=0.90, similarity_method = 'cosine'):
    """Finds pairs of similar images based on feature similarity.

    Args:
        features: Dictionary of image paths and their extracted feature vectors.
        threshold: Minimum similarity score to consider images as similar.
        similarity_method: The method used to calculate similarity. Options are 'cosine'

    Returns:
       A list of tuples containing (image1_path, image2_path, similarity_score).
    """
    similar_pairs = []
    image_paths = list(features.keys())
    feature_vectors = np.array(list(features.values()))
    if similarity_method == 'cosine':
       feature_vectors = normalize(feature_vectors)
       similarity_matrix = cosine_similarity(feature_vectors)
    else:
       raise ValueError("Invalid Similarity Method")


    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            similarity_score = similarity_matrix[i, j]

            if similarity_score > threshold:
                similar_pairs.append((image_paths[i], image_paths[j], similarity_score))
    return similar_pairs
def extract_feature(image):
    img_gray = np.array(image.convert('L')) # Convert to grayscale
    hog_features = hog(img_gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) # Adjust parameters if needed
    return hog_features
def calculate_similarity(image1, image2):
    feature1 = extract_feature(image1)
    feature2 = extract_feature(image2)
    feature_vectors = np.array([feature1, feature2])
    feature_vectors = normalize(feature_vectors)
    similarity_matrix = cosine_similarity(feature_vectors)
    return similarity_matrix[0][1]

def number_to_string5(number):
    """Converts a number to a 5-character string, padding with leading zeros if necessary."""
    return f"{number:05}"

if __name__ == "__main__":
    path = "data/218_EMO-1_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/images/"
    x = 28
    path1 = path + number_to_string5(x) + "_06.png"
    image1 = Image.open(path1).convert("RGB")

    for i in range(x+1, 65):
        # path1 = path + number_to_string5(i) + "_06.png"
        path2 = path + number_to_string5(i) + "_06.png"
        # image1 = Image.open(path1).convert("RGB")
        image2 = Image.open(path2).convert("RGB")
        sim = calculate_similarity(image1, image2)
        print(i," ", sim)
#    image_directory = "data/218_EMO-1_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/images"  # Replace with your image directory
#    images = load_images(image_directory)

#    if not images:
#      print("No images were loaded. Exiting.")
#      exit()


#    feature_type_choice = 'hog'  # options: 'hog', 'pixel'
#    features = extract_features(images, feature_type=feature_type_choice)

#    similarity_threshold = 0.85 # Adjust
#    similarity_method_choice = 'cosine' # Options: 'cosine'
#    similar_image_pairs = find_similar_images(features, threshold=similarity_threshold, similarity_method=similarity_method_choice)

#    if not similar_image_pairs:
#         print("No similar images found based on threshold. Try lowering the threshold or using a different feature extraction method.")
#         exit()

#    print("Similar Image Pairs (Path1, Path2, Similarity):")
#    for path1, path2, score in similar_image_pairs:
#       print(f"{path1}, {path2}, Similarity: {score:.4f}")