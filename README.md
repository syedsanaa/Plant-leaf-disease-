Description:
This is a work-in-progress project that involves integrating the SAM2-Unet encoder into the Planet model. Due to high memory requirements, the project is not yet complete and is still under development. The model structure, dataset preprocessing, and evaluation files are present, but some parts may need optimization to handle memory constraints.

Files Overview
SAM2-Planet.py
This file contains the model structure. It integrates the encoder from the SAM2-Unet model into the Planet model architecture. Currently, the integration is in progress due to memory limitations.

Dataset Preprocessing
This file handles the preprocessing of the dataset to make it compatible with the model's input requirements. Preprocessing has been completed, but further optimizations may be needed for larger datasets.

Evaluation Files
There are separate files for training and testing:

Train: Handles the training pipeline and model fitting. It is under development and may face memory issues with larger datasets.
Test: Evaluates the performance of the trained model on test data. This file is functional but may require further testing and optimization.
How It Works
SAM2-Planet.py
The main structure is in SAM2-Planet.py, where we extract the encoder part from the SAM2-Unet and integrate it into the Planet model. Due to memory constraints, this part of the project is still being optimized for efficiency.

Dataset
The dataset preprocessing is performed in the dataset file, ensuring that the data is cleaned and formatted before training. Larger datasets may need further preprocessing to reduce memory usage.

Training & Testing
The training file is responsible for training the model, while the testing file evaluates the model's performance. Both are still in development, with some parts facing memory issues during large-scale testing.
