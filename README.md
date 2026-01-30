Real-Time Video Captioning

This repository implements a real-time video captioning system using deep learning techniques. The system processes video input and generates descriptive captions for video frames or segments.

Overview

The project contains code for training models, preparing datasets, and running inference for video captioning. Multiple scripts and a Jupyter notebook are included for experimentation.

Repository Structure
Real-time-video-captioning/
├── README.md
├── requirements.txt
├── main.py
├── main_1.py
├── main_2.py
├── gan_nl.py
├── chk.py
├── prepare_flickr8k.py
├── train (1).ipynb
├── discriminator_now.pth
├── generator_now.pth

Installation

Clone the repository:

git clone https://github.com/Ramkumar64/Real-time-video-captioning
cd Real-time-video-captioning


Install dependencies:

pip install -r requirements.txt

Usage
Running the Model

To run the captioning system, execute one of the main scripts. For example:

python main.py


Replace main.py with main_1.py or main_2.py depending on the variant you want to run.

Jupyter Notebook

The train (1).ipynb notebook is provided for interactive exploration, training, and evaluation.

Dataset Preparation

If you are preparing your own dataset such as Flickr8k, use the provided script:

python prepare_flickr8k.py


Modify dataset paths in the script as needed.

Training

Training scripts can be used to train models from scratch. Ensure the required dataset and environment are properly configured before running.

Model Weights

Pretrained model weights are included in the repository:

discriminator_now.pth

generator_now.pth

These can be used for inference without training from scratch.

Requirements

All dependencies are listed in requirements.txt. Install them using:

pip install -r requirements.txt

Notes

Replace dataset paths in the code before running.

GPU acceleration is recommended for training and real-time inference.

Adjust parameters and paths in the scripts as per your setup.

License

Specify your project license here (example: MIT License).

Contribution

Contributions, issues, and suggestions are welcome. Add details for how you prefer contributions to be made (e.g., via pull requests).
