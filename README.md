# Multi-Channels Echo State Extreme Learning Machine (Multi-ESN-EEWS)

This repository contains the implementation of the Multi Echo-State Extreme Learning Machine (Multi ES-ELM), an efficient and effective model designed for predicting earthquake strong motions in real-time Earthquake Early Warning (EEW) systems[cite: 1]. 

## Overview
Predicting strong ground motions accurately and quickly is critical for mitigating seismic risks[cite: 1]. While conventional deep learning models (like CNNs and RNNs) offer high precision, they require substantial computational resources, limiting their deployment in environments with restricted processing capabilities[cite: 1]. 

This project proposes a lightweight alternative that utilizes multi-channel time-series data (NS, EW, and Z axes of initial P-waves) to predict Peak Ground Acceleration (PGA)[cite: 1]. The model architecture consists of two main parts: an Echo State Network (ESN) layer with fixed weights for rapid feature extraction without backpropagation, and an Extreme Learning Machine (ELM) trained using a pseudo-inverse method for rapid classification[cite: 1].

## Key Features
*   **Highly Lightweight:** The model requires only 882 parameters and occupies just 0.003 MB of storage space[cite: 1].
*   **Ultra-Fast Training:** The training process is completed in approximately 1.92 seconds[cite: 1].
*   **Memory Efficient:** The model consumes significantly lower memory (20.28 GB) during training compared to standard CNNs (328.75 GB) and RNNs (169.64 GB)[cite: 1].
*   **High Recall Performance:** It delivers an outstanding recall of 96.50 ± 0.52, ensuring that hazardous seismic waves are detected effectively to minimize missed warnings[cite: 1].
*   **Competitive Accuracy:** The model achieves an overall accuracy of 93.46 ± 0.22, a precision of 81.53 ± 0.53, and an F1-score of 88.38 ± 0.37[cite: 1].

## Data Structure
The input to the Multi ES-ELM model is multivariate time series data from seismic station recordings[cite: 1]. It utilizes the initial 5 seconds of the P-wave (500 samples) across three channels (north-south, east-west, and up-down)[cite: 1]. 

## Citation
If you use this code or model in your research, please cite the following paper:

> Chomchit, P., Aramkul, S., Somchit, Y., & Champrasert, P. (2025). Earthquake early warning using multi-channels echo state extreme learning machine. *Journal of Current Science and Technology, 15*(4), Article 139. https://doi.org/10.59796/jcst.V15N4.2025.139[cite: 1]
