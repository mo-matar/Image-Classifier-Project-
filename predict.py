import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
from PIL import Image
import warnings
import logging
import tf_keras

warnings.filterwarnings('ignore')
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def process_image(image):
    """Process an image into a format suitable for the model."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image

def predict(image_path, model, top_k):
    """Predict the top K classes for the image using the model."""
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_probabilities = predictions[0][top_indices]
    top_classes = [str(i) for i in top_indices]
    return top_probabilities, top_classes

def display_percentage_bar(percentage):
    length = int(percentage)  
    return "#" * length


def main(args):

    #load the model
    model = tf_keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    #predict the top classes
    top_k = args.top_k
    probs, classes = predict(args.image_path, model, top_k)

    #map class indices to names if --category_names is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        class_labels = [class_names.get(str(cls), f"Class {cls}") for cls in classes]
    else:
        class_labels = classes  

    
    print(f"Predictions for image: {args.image_path}")
    for i in range(len(probs)):
        percentage = probs[i] * 100
        bar = display_percentage_bar(percentage)
        print(f"{i+1}: {class_labels[i]} with probability {percentage:.2f}%")
        print(f"{bar}\n")

if __name__ == "__main__":
    #set up the command-line args
    parser = argparse.ArgumentParser(description="A Flower Image Classifier Application.\nUsage example:\npredict.py [image_path.jpg] [model.h5] [--top_k 3] [--category_names MAP_FILE.json]")
    parser.add_argument("image_path", help="(Required) The path to the flower image to be classified.")
    parser.add_argument("model_path", help="(Required) The path to the pre-trained .h5 model file.")
    parser.add_argument("--top_k", action="store", dest="top_k", type=int, default=5, help="Return the top K classes for the image. Default is 5.")
    parser.add_argument("--category_names", action="store", dest="category_names", help="Path to the JSON file mapping labels to class names.")
    
    args = parser.parse_args()
    main(args)
