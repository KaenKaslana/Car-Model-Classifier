# Car-Model-Classifier

A PyTorch-based car model classifier using a VGG16 model trained on various Honda car models. This repository includes scripts for training the model, saving and loading model weights, and predicting car models from images.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains the implementation of a car model classifier using a VGG16 neural network. The classifier is trained on images of various Honda car models and is capable of predicting the model of a car given an image.

## Features

- Preprocessing and augmentation of car images.
- Training script to train the VGG16 model.
- Script to save and load trained model weights.
- Inference script to predict car models from new images.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/Car-Model-Classifier.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Car-Model-Classifier
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Model Training

To train the model, prepare your dataset and run the training script:

1. Place your dataset and annotation file in the appropriate directories.
2. Modify the paths in the training script if necessary.
3. Run the training script:
    ```sh
    python train.py
    ```

### Model Inference

To use the trained model for prediction:

1. Ensure you have the trained model weights saved as `Car_10.pth` or the appropriate epoch file.
2. Use the inference script to predict the car model from an image:
    ```sh
    python predict.py --image_path path/to/your/image.jpg
    ```

## Model Training

Detailed instructions on how to train the model using your dataset. This includes preprocessing steps, data augmentation, and training parameters.

## Model Inference

Detailed instructions on how to use the trained model to predict car models from new images. This section explains how to load the model and perform inference on single or multiple images.

## Future Work

This is version 1.0 of the Car-Model-Classifier. Future updates will include:
- Adding more car models to the dataset.
- Improving the model architecture for better accuracy.
- Implementing a web interface for easy image upload and model prediction.
- Adding more data augmentation techniques to improve model robustness.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
