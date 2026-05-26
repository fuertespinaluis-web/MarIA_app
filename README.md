# MarIA: MALDI-TOF Identification Application

MarIA is an open-source application and end-to-end machine learning pipeline for species-level identification within the *Streptococcus mitis* group (SmG) using MALDI-TOF mass spectrometry data.


## Overview

Accurate discrimination between *Streptococcus pneumoniae* and closely related members of the SmG remains challenging using conventional MALDI-TOF workflows. MarIA addresses this limitation by combining standardized spectral preprocessing with machine learning to improve species-level identification.

The project includes:
- A complete preprocessing and modeling pipeline
- A trained Random Forest classifier
- An interactive application for prediction and confidence visualization


## Features

- ✅ Standardized MALDI-TOF preprocessing pipeline  
- ✅ Random Forest-based classification model  
- ✅ Cross-validation and threshold calibration  
- ✅ Support for external (out-of-distribution) validation  
- ✅ Confidence visualization module  
- ✅ User-friendly interface for spectrum analysis  


## Machine Learning Pipeline

The pipeline consists of the following steps:

1. **Preprocessing**
   - Baseline correction
   - Smoothing
   - Peak detection / binning
   - Feature vector generation

2. **Training**
   - Random Forest classifier
   - Hyperparameter tuning via GridSearch (5-fold cross-validation)
   - Class imbalance handling (oversampling / augmentation)

3. **Threshold Definition**
   - Initial threshold derived from cross-validation
   - Recalibration using external (OOD) data

4. **Evaluation**
   - In-distribution (ID) test set
   - Out-of-distribution (OOD) validation
   - Performance metrics (accuracy, sensitivity, specificity, etc.)

5. **Deployment**
   - Integration into the MarIA application
   - Interactive prediction and confidence assessment


## Application (MarIA)

The MarIA application allows users to:

- Upload MALDI-TOF spectra  
- Apply standardized preprocessing  
- Obtain species-level predictions  
- Visualize prediction confidence in the context of external validation data  


## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/maria-malditof.git
cd maria-malditof
