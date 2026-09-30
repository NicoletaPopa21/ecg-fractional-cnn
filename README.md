# ecg-fractional-cnn
Advanced Deep Learning for ECG Arrhythmia Detection using Fractional Calculus

The project uses Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification: A Deep Transferable Representation." arXiv preprint arXiv:1805.00794 (2018).
This repository implements a modular Deep Learning pipeline for detecting heart arrhythmias from ECG signals (MIT-BIH dataset). It utilizes 1D Convolutional Neural Networks (CNNs) enhanced by mathematical fractional calculus algorithms (Caputo and Grünwald-Letnikov derivatives) to amplify pathological signal features.

**Key Features & Performance**
Multichannel Feature Fusion: Stacks raw ECG signals with GL and Caputo fractional derivatives to create a robust 3-channel input.

Advanced Architecture: Custom 1D CNN with Global Average Pooling, Dropout optimization, and dynamic Learning Rate decay.

High Accuracy: Achieved 96.18% accuracy in distinguishing between Normal, Supraventricular, Ventricular, Fusion, and Unknown heartbeats.

**Mathematical Signal Processing**
Standard ECG analysis often struggles with subtle morphological changes in pathological beats. By applying fractional derivatives, we mathematically amplify these anomalies before feeding them into the neural network:

Grünwald-Letnikov (GL): Highlights finite differences in the signal.

Caputo: Provides a continuous fractional response, emphasizing the active pathological zones.

<img width="691" height="484" alt="grafic - Copie" src="https://github.com/user-attachments/assets/c69c22b5-c84e-404e-99b2-9cf76c5226c9" />
<img width="689" height="484" alt="grafic" src="https://github.com/user-attachments/assets/643a7ca1-32a4-4449-8cd0-b24c5cfc2093" />


**Modular Architecture**
The codebase is designed to be modular and scalable, allowing for easy testing of different mathematical filters or simplified model architectures:

src/preprocessing.py: Contains the signal filtering logic (scipy.signal).

src/models.py: Defines the Keras functional API models.

train.py: Handles the training loop, Early Stopping, and checkpointing.

**Results & Convergence**
The fusion model demonstrates stable convergence without overfitting, heavily benefiting from the dropout layers and ReduceLROnPlateau callbacks.
