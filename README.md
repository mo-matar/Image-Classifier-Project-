# Udacity Machine Learning Nanodegree Program Project: Image Classifier with Deep Learning

This project implements an image classifier using deep learning with TensorFlow. The classifier is trained on a flower dataset and converted into a command-line application that predicts the class of an input image.

## Project Overview

The project is divided into two main parts:

### Part 1: Developing the Image Classifier
- Implemented in a Jupyter notebook using TensorFlow.
- Trains a deep neural network on the flower dataset.
- Saves the trained model as a checkpoint for further use.
- Includes techniques to reduce the size of the model checkpoint by using deeper but narrower networks.

### Part 2: Building the Command Line Application
- A Python script `predict.py` that:
  - Accepts an input image and a saved model to make predictions.
  - Outputs the most probable class (flower name) along with probabilities.
  - Includes optional arguments for top K predictions and category mappings.

## Features

- Predict the class of an input image using a trained model.
- Display the top K most likely classes and their probabilities.
- Map numerical labels to flower names using a JSON file.

## Files in the Repository

- **python Notebook**:  
  Contains the implementation of the image classifier and training process.
- **`predict.py`**:  
  Command-line application for making predictions.
- **Test Images**:  
  A folder `./test_images/` with four flower images for testing:
  - `cautleya_spicata.jpg`
  - `hard-leaved_pocket_orchid.jpg`
  - `orange_dahlia.jpg`
  - `wild_pansy.jpg`

## Instructions for Running the Project

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies (TensorFlow, NumPy, Matplotlib, argparse).
3. Train the model using the Jupyter notebook and save the checkpoint.
4. Use the `predict.py` script to make predictions.
## Command Line Usage
- Basic Usage
```bash
$ python predict.py /path/to/image saved_model
```
- Options
  1. top k most likely classes:
  ```bash
  $ python predict.py /path/to/image saved_model --top_k K
  ```
  2. Category Names: Map the names using a `JSON` file
  ```bash
  $ python predict.py /path/to/image saved_model --category_names map.json
  ```
  3. Example Commands:
  ```bash
  # Basic prediction
  $ python predict.py ./test_images/orchid.jpg my_model.h5
  
  # Top 3 predictions
  $ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
  
  # Mapping labels to flower names
  $ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
  ```
  
## Notes
- Ensure TensorFlow 2.0 or above is installed for compatibility.
- Optimize the size of the saved checkpoint to ensure efficient storage and loading.
- Refer to the rubric for detailed project requirements and grading criteria.


---
