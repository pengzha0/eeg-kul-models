ASA EEG Database README

This repository provides an EEG database and the corresponding source code for auditory spatial attention decoding (ASAD) research.

**Contents:**
- 64-channel EEG data: responses to two-speaker speech stimuli from 10 different locations (±90°, ±60°, ±45°, ±30°, and ±5°).
- Baseline model (CA-CNN) for ASAD, along with other models.
- Preprocessing, training and testing code for model evaluation.
- Visualization script for the dataset paper.

**For More Information:**
Refer to the paper "ASA: An Auditory Spatial Attention Dataset with Multiple Speaking Locations" for a detailed description of the dataset.

**Setup Guide:**
1. Download and unzip all subjects' EEG data in your asa_data folder.
2. Modify the path settings in "main.py" to align with your local setup.
3. Verify that your Python environment meets the following requirements:
   - Python 3.10.8
   - TensorFlow 2.13.0
   - MNE 1.5.0
   - NumPy 1.23.5
   - SciPy 1.10.1
   - Matplotlib 3.7.1
4. Execute "main.py" to initiate the analysis process.

**Results:**
The analysis results will be saved as "results_***.txt" and "averages_***.txt" files.