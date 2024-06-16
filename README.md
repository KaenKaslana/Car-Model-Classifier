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

**Download the Dataset**:
    - The dataset used for training is not included in this repository due to size constraints.
    - You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/occultainsights/honda-cars-over-11k-labeled-images?resource=download).
    
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
2. Place the image you want to predict in the `Put_your_Image_here` folder.
3. Use the inference script to predict the car model from an image:
    ```sh
    python Car_Detector.py --image_path Put_your_Image_here/image.jpg
    ```

### Generate Annotation File

To generate the annotation file required for training:

1. Run the `txt.py` script:
    ```sh
    python txt.py
    ```
   This will create a `cls_honda_cars.txt` file with labels and paths for images in the `honda_cars` dataset.

## Model Training

Detailed instructions on how to train the model using your dataset. This includes preprocessing steps, data augmentation, and training parameters.

#### Step 1: Prepare Your Dataset

1. **Organize your dataset**:
    - Ensure your dataset is organized in folders, where each folder is named after the class label it contains.
    - Place all class folders in a parent directory (e.g., `honda_cars`).

    ```
    honda_cars/
    ├── accord/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── amaze/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── ...
    └── vezel/
        ├── img1.jpg
        ├── img2.jpg
        └── ...
    ```

2. **Generate the annotation file**:
    - Use the `txt.py` script to create an annotation file that lists the paths and labels of all images.

    ```sh
    python txt.py
    ```

    This will generate a `cls_honda_cars.txt` file.

#### Step 2: Preprocessing and Data Augmentation

The `DataGenerator` class in `data.py` handles preprocessing and data augmentation. Here are the main steps:

1. **Convert images to RGB**:
    - Ensure all images are in RGB format using the `cvtColor` function.

    ```python
    def cvtColor(image):
        if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image
    ```

2. **Resize images and apply random transformations**:
    - Resize images to the input shape (224x224) and apply random transformations such as flipping and color jitter.

    ```python
    def get_random_data(image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        # Resize and augment images
        # Apply random transformations
    ```

3. **Normalize images**:
    - Normalize image pixel values to the range [-1, 1].

    ```python
    def preprocess_input(x):
        x /= 127.5
        x -= 1.
        return x
    ```

#### Step 3: Training Parameters

1. **Define the model**:
    - Use the `vgg16` function from `net.py` to create a VGG16 model with pretrained weights.

    ```python
    from net import vgg16

    net = vgg16(pretrained=True, progress=True, num_classes=17)
    ```

2. **Set up the optimizer and learning rate scheduler**:
    - Use the Adam optimizer with a learning rate of 0.0001.
    - Use a StepLR scheduler to adjust the learning rate.

    ```python
    import torch.optim as optim

    lr = 0.0001
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    ```

3. **Create data loaders**:
    - Use `DataLoader` to create training and validation data loaders with a batch size of 128.

    ```python
    from torch.utils.data import DataLoader

    train_data = DataGenerator(train_lines, input_shape, True)
    val_data = DataGenerator(val_lines, input_shape, False)

    batch_size = 128
    gen_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    gen_val = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    ```

4. **Train the model**:
    - Train the model for 10 epochs, saving the model weights after each epoch.

    ```python
    epochs = 10
    for epoch in range(epochs):
        net.train()
        total_train_loss = 0
        for data in gen_train:
            img, label = data
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = net(img)
            loss = nn.CrossEntropyLoss()(outputs, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()

        total_val_loss = 0
        total_correct = 0
        net.eval()
        with torch.no_grad():
            for data in gen_val:
                img, label = data
                img, label = img.to(device), label.to(device)
                outputs = net(img)
                loss = nn.CrossEntropyLoss()(outputs, label)
                total_val_loss += loss.item()
                total_correct += (outputs.argmax(1) == label).sum().item()

        avg_train_loss = total_train_loss / len(gen_train)
        avg_val_loss = total_val_loss / len(gen_val)
        accuracy = total_correct / len(val_data)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Accuracy: {accuracy:.2%}')
        torch.save(net.state_dict(), f'Trained_Parameters/Car_{epoch+1}.pth')
        print(f'Model saved for epoch {epoch+1}')
    ```

By following these steps, you can train the VGG16 model on your dataset and save the trained model weights.

## Model Inference

Detailed instructions on how to use the trained model to predict car models from new images. This section explains how to load the model and perform inference on single or multiple images.

To use the trained model to predict car models from new images, follow these steps:

1. **Place the image for prediction**:
    - Place the image you want to predict in the `Put_your_Image_here` folder and name it `image.jpg`.

2. **Run the inference script**:
    - Use the `Car_Detector.py` script to load the trained model and perform inference on the image.

    ```sh
    python Car_Detector.py --image_path Put_your_Image_here/image.jpg
    ```

#### Example `Car_Detector.py`

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from net import vgg16  # Ensure your net.py contains vgg16 definition
import os

classes = ['accord', 'amaze', 'brio', 'city', 'civic', 'clarity', 'freed', 'insight', 'legend', 'mobilo', 'nsx', 'odyssey', 'passport', 'pilot', 'ridgeline', 's660', 'vezel']

def load_model(model_path, num_classes=17):
    model = vgg16(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return classes[preds]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='Put_your_Image_here/image.jpg', help='Path to the image')
    args = parser.parse_args()
    
    model_path = 'Trained_Parameters/Car_10.pth'
    model = load_model(model_path)
    prediction = predict(model, args.image_path)
    print(f'The predicted class is: {prediction}')
```
This script will load the trained model, preprocess the input image, and print the predicted car model.

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
