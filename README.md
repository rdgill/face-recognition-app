# Face Recognition App

This project develops a face recognition system using DeepFace VGG-Face embeddings, PCA, and SVM to classify individuals from images, achieving 96.05% accuracy. Completed during my Postgraduate Program in Deep Learning & AI (Grade: A, 2025), it draws on my 12+ years at a PSU bank, where I used AI analytics dashboards for fraud prevention and 46% commission growth.

## Motivation
At a PSU bank, I implemented AI-driven fraud alerts for 500+ daily transactions, inspiring this project to explore computer vision for identity verification. It supports NUS’s AI research in vision-based security and applications at AI firms like AWS or Grab, aligning with my PhD and career goals.

## Methods
- **Dataset**: PINS dataset (10,770 unique images, ~100 celebrities, .jpg/.png). Original PGP data is proprietary; sample images are in `data/sample_images/`.
- **Preprocessing**: Extracted unique images from PINS.zip, generated metadata (file path, person label), and computed 4096D VGG-Face embeddings using DeepFace in batches of 500.
- **Feature Reduction**: Applied PCA (50 components) to reduce embedding dimensionality.
- **Model**: Trained a linear SVM classifier on PCA-transformed embeddings with an 80-20 train-test split.
- **Tools**: Python, DeepFace, TensorFlow, OpenCV, Scikit-learn, Matplotlib, Pandas, Google Colab.

## Results
- Linear SVM classifier achieved **96.05% accuracy** on the test set.
- Correctly identified test images (e.g., Benedict Cumberbatch, Dwayne Johnson) with precise predictions.
- Visualized PCA embeddings and sample predictions (see `figures/`).
- Saved PCA and SVM models for deployment (`models/pca.pkl`, `models/svm.pkl`).

## Relevance
This project extends my PSU bank experience in AI-driven fraud prevention to computer vision, aligning with NUS’s AI vision research (e.g., biometric authentication) and industry needs in security (e.g., AWS Rekognition, Grab’s KYC). It showcases deep learning and classification skills for PhD research and AI firm roles.

## Usage
1. Clone the repo:
   ```bash
   git clone https://github.com/ravdeepgill/face-recognition-app.git
