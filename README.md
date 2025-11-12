# Marine-Animal-Sound-Recognition-VGG15-Flask-
This project uses a deep learning model based on the **VGG15** architecture to classify and recognize different species of marine animals (such as whales, dolphins, etc.) from their vocalizations (reconnaissance_audio_animaux_marrins(1).py).
A simple web application built with **Flask** is also included to allow testing the trained model by uploading an audio file (Flask).

## Features

* **Audio Processing**: Conversion of raw audio files into spectrograms (or MFCCs) for analysis.

* **VGG15 Model**: Use of the (pre-trained) VGG15 architecture for sound classification.

* **Training**: Script to train or fine-tune the model on a dataset of marine sounds.

* **Deployment API**: A Flask application that exposes an endpoint to predict species from a new audio file.

## Technologies Used

Flask
tensorflow== 2.16.2
librosa== 0.11.0
numpy== 1.26.4
scikit-image==0.22.0
matplotlib== 3.10.6

## Installation

1. **Clone this repository:**

2. **Create a virtual environment (recommended):**

3. **Install the dependencies:**
```bash
pip install -r requirements.txt

```

4. **Download the data and model:**

* **Data:** The audio dataset is not included due to its size. You must download the **Watkins Marine Mammal Sound Database** dataset.

* **Model:** The trained model is not included due to size limitations. You must train the VGG 15 model using the provided file. The model is saved automatically after training.

## Usage

### 1. First, run the training file to generate the trained model to be used in the app.py file of your Flash project. After saving the model, verify the paths and names in app.py.
