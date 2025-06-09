To implement the **Multimodal AI-Driven Framework for Non-Invasive Fetal Cardiac Health Monitoring and Stress Detection** using the NInFEA dataset in Google Colab, we’ll create a step-by-step Python-based pipeline that leverages the dataset’s 27-channel fetal ECG (fECG), pulsed-wave Doppler (PWD), and maternal respiration signals. The code will cover data loading, preprocessing, fECG extraction, feature extraction, multimodal fusion, machine learning, and explainable AI, tailored for beginners. I’ll break it down into clear, manageable steps, explaining each part in detail, including how to set up the environment, handle the dataset, and execute the pipeline. Since Google Colab is a cloud-based Python environment, we’ll ensure all dependencies are installed and the code is compatible with its setup.

### Prerequisites
- **Google Colab Account**: we need a Google account to access Google Colab (https://colab.research.google.com/). No prior setup is required, as Colab provides a free Python environment with GPU/TPU support.
- **NInFEA Dataset**: The dataset is available on PhysioNet (2.0 GB uncompressed). We’ll download it directly in Colab.
- **Basic Understanding**: No prior coding knowledge is assumed. I’ll explain each step, including how to run cells in Colab and interpret outputs.

### Step-by-Step Implementation

#### Step 1: Set Up Google Colab Environment
**What**: Install necessary Python libraries and set up the workspace in Google Colab.
**Why**: Colab doesn’t have all required libraries (e.g., for signal processing, machine learning, or explainable AI) pre-installed, so we need to install them. We’ll also create a directory to store the NInFEA dataset.
**How**:
1. Open Google Colab:
   - Go to https://colab.research.google.com/.
   - Click “New Notebook” to create a new notebook.
2. Run the following code in a new code cell to install dependencies and create a directory for the dataset.

```python
# Step 1: Install dependencies and set up directory
# Explanation: 
# - '!' runs shell commands in Colab.
# - We install libraries: scipy (signal processing), numpy (arrays), pandas (data handling),
#   matplotlib/seaborn (plotting), tensorflow (machine learning), scikit-learn (ML tools),
#   wfdb (PhysioNet data), opencv-python (image processing for PWD), and shap/lime (explainable AI).
# - 'mkdir' creates a directory to store the NInFEA dataset.

!pip install scipy numpy pandas matplotlib seaborn tensorflow scikit-learn wfdb opencv-python shap lime
import os
os.makedirs('ninfea_data', exist_ok=True)
print("Dependencies installed and directory created!")
```

**What to Do**:
- Copy the code above into a new code cell in Colab.
- Click the “Play” button (triangle) to the left of the cell to run it.
- Wait for the output to confirm installation (may take 1–2 minutes). we’ll see “Dependencies installed and directory created!” if successful.
- **Tip**: If we see errors (e.g., “pip not found”), ensure we’re connected to a runtime (click “Connect” at the top right of Colab).

#### Step 2: Download and Extract the NInFEA Dataset
**What**: Download the NInFEA dataset from PhysioNet and extract it in Colab.
**Why**: The dataset contains the 27-channel fECG, PWD images, and respiration signals we need for analysis. Colab’s temporary storage requires us to download and extract it each session.
**How**:
1. Run the following code to download the dataset using `wget` and extract the ZIP file.

```python
# Step 2: Download and extract NInFEA dataset
# Explanation:
# - 'wget' downloads the ZIP file from PhysioNet (792.9 MB).
# - 'unzip' extracts it to the 'ninfea_data' directory.
# - We list the directory contents to verify extraction.

!wget -r -N -c -np https://physionet.org/files/ninfea/1.0.0/ -P ninfea_data
!unzip ninfea_data/physionet.org/files/ninfea/1.0.0/ninfea-1.0.0.zip -d ninfea_data
!ls ninfea_data
```

**What to Do**:
- Create a new code cell in Colab and paste the code above.
- Run the cell by clicking the “Play” button.
- **Expected Output**: The download may take 5–10 minutes depending on the internet speed. After extraction, we’ll see a list of folders (e.g., `bin_format_ecg_and_respiration`, `pwd_images`, `wfdb_format_ecg_and_respiration`) and files (`LICENSE.txt`, `RECORDS`, `SHA256SUMS.txt`).
- **Tip**: If the download fails, check the Colab runtime (reconnect if disconnected) or try running the cell again.

#### Step 3: Load and Preprocess NInFEA Data
**What**: Load the electrophysiological (.bin) and PWD (.bmp) files using the provided Matlab tools, adapted for Python.
**Why**: The NInFEA dataset provides .bin files for fECG and respiration and .bmp files for PWD. We need to read these into Python for processing, using the dataset’s structure (27 channels, maternal respiration, etc.).
**How**:
1. Since NInFEA provides Matlab tools, we’ll use Python equivalents (`scipy` for signal processing, `wfdb` for PhysioNet data, `cv2` for PWD images).
2. Run the following code to load a sample .bin file and a corresponding PWD image, and preprocess the signals.

```python
# Step 3: Load and preprocess NInFEA data
# Explanation:
# - We use 'numpy' to read .bin files (based on NInFEA's IEEE little-endian format).
# - We load a sample .bin file (electrophysiological data) and .bmp file (PWD image).
# - We extract fECG (channels 1–24), maternal ECG (channels 25–27), and respiration (channel 32).
# - We preprocess signals by normalizing and filtering noise.

import numpy as np
import scipy.io
import scipy.signal
import cv2
import os

# Define paths to sample files (adjust based on actual file names after extraction)
bin_path = 'ninfea_data/bin_format_ecg_and_respiration/sample.bin'  # Replace with actual file
pwd_path = 'ninfea_data/pwd_images/sample.bmp'  # Replace with actual file

# Function to read .bin file (based on NInFEA format: 8-byte double precision)
def read_bin_file(file_path):
    with open(file_path, 'rb') as f:
        # Read header: sampling frequency, # channels, # samples
        fs = np.fromfile(f, dtype=np.float64, count=1)[0]  # Sampling frequency
        n_channels = np.fromfile(f, dtype=np.uint64, count=1)[0]  # # channels
        n_samples = np.fromfile(f, dtype=np.uint64, count=1)[0]  # # samples
        # Read data: 8 bytes per sample, all channels
        data = np.fromfile(f, dtype=np.float64).reshape(n_samples, n_channels)
    return data, fs

# Load electrophysiological data
data, fs = read_bin_file(bin_path)
fECG = data[:, :24]  # Channels 1–24 (unipolar, fECG)
maternal_ECG = data[:, 24:27]  # Channels 25–27 (maternal ECG)
respiration = data[:, 31]  # Channel 32 (respiration)

# Preprocess signals: Normalize and apply bandpass filter (0.5–100 Hz for ECG)
fECG = (fECG - np.mean(fECG, axis=0)) / np.std(fECG, axis=0)  # Normalize
b, a = scipy.signal.butter(4, [0.5, 100], btype='bandpass', fs=fs)
fECG = scipy.signal.filtfilt(b, a, fECG, axis=0)

# Load PWD image
pwd_image = cv2.imread(pwd_path, cv2.IMREAD_GRAYSCALE)

# Simple PWD envelope extraction (approximation of Matlab tool)
def extract_pwd_envelope(image):
    # Convert image to 1D signal by averaging columns
    signal = np.mean(image, axis=0)
    # Smooth signal to approximate envelope
    smoothed = scipy.signal.savgol_filter(signal, window_length=51, polyorder=3)
    return smoothed

pwd_envelope = extract_pwd_envelope(pwd_image)

print("fECG shape:", fECG.shape)
print("Maternal ECG shape:", maternal_ECG.shape)
print("Respiration shape:", respiration.shape)
print("PWD envelope shape:", pwd_envelope.shape)
```

**What to Do**:
- Create a new code cell and paste the code above.
- **Replace File Paths**: After running Step 2, check the `ninfea_data` directory for actual .bin and .bmp file names (e.g., `s01.bin`, `s01.bmp`). Update `bin_path` and `pwd_path` accordingly.
- Run the cell. It will load one sample record, preprocess the fECG, and extract a PWD envelope.
- **Expected Output**: Shapes of the loaded data, e.g., `fECG shape: (n_samples, 24)`, indicating the number of samples and channels.
- **Tip**: If the file paths are incorrect, we’ll get a “File not found” error. Use `!ls ninfea_data/bin_format_ecg_and_respiration` to list available .bin files and update the path.

#### Step 4: Fetal ECG Extraction Using ICA
**What**: Apply Independent Component Analysis (ICA) to extract clean fECG from the 27-channel data.
**Why**: fECG signals are mixed with maternal ECG and noise. ICA separates these sources, leveraging NInFEA’s high channel count.
**How**:
1. Use `scikit-learn`’s ICA implementation to separate fECG from maternal ECG and noise.
2. Run the following code to apply ICA and select the fECG component.

```python
# Step 4: Extract fECG using ICA
# Explanation:
# - We use FastICA from scikit-learn to separate fECG from maternal ECG and noise.
# - We assume the fECG component has the highest correlation with PWD-derived heartbeats.
# - We use the PWD envelope as a reference to select the correct ICA component.

from sklearn.decomposition import FastICA
import numpy as np

# Apply ICA to fECG channels
ica = FastICA(n_components=24, random_state=42)
ica_components = ica.fit_transform(fECG)

# Select fECG component by correlating with PWD envelope
# Resample PWD envelope to match fECG sampling rate (2048 Hz)
pwd_resampled = scipy.signal.resample(pwd_envelope, fECG.shape[0])
correlations = [np.corrcoef(ica_components[:, i], pwd_resampled)[0, 1] for i in range(ica_components.shape[1])]
fECG_component_idx = np.argmax(np.abs(correlations))
clean_fECG = ica_components[:, fECG_component_idx]

print("Clean fECG shape:", clean_fECG.shape)
```

**What to Do**:
- Create a new code cell and paste the code above.
- Run the cell after Step 3. It assumes `fECG` and `pwd_envelope` are defined from Step 3.
- **Expected Output**: Shape of the clean fECG signal, e.g., `Clean fECG shape: (n_samples,)`.
- **Tip**: If ICA fails (e.g., due to low signal quality), try adjusting `n_components` or preprocessing (e.g., stricter bandpass filter).

#### Step 5: Feature Extraction
**What**: Extract features from clean fECG, PWD, and respiration for machine learning.
**Why**: Features like RR intervals (fECG), atrioventricular intervals (PWD), and HRV (respiration) capture cardiac and stress patterns.
**How**:
1. Extract RR intervals from fECG, atrioventricular intervals from PWD, and HRV from respiration.
2. Run the following code to compute these features.

```python
# Step 5: Extract features from fECG, PWD, and respiration
# Explanation:
# - For fECG: Detect R-peaks and compute RR intervals.
# - For PWD: Compute atrioventricular intervals from envelope peaks.
# - For respiration: Compute HRV-like features (standard deviation of respiration signal).
# - We assume clean_fECG, pwd_envelope, and respiration are from previous steps.

import scipy.signal
import numpy as np

# fECG: Detect R-peaks and compute RR intervals
r_peaks, _ = scipy.signal.find_peaks(clean_fECG, distance=int(fs*0.2))  # Min 200 ms between peaks
rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds
rr_mean = np.mean(rr_intervals)
rr_std = np.std(rr_intervals)

# PWD: Detect peaks in envelope for atrioventricular intervals
pwd_peaks, _ = scipy.signal.find_peaks(pwd_envelope, distance=50)  # Adjust based on PWD resolution
av_intervals = np.diff(pwd_peaks) / 60  # Convert to seconds (60 Hz PWD)
av_mean = np.mean(av_intervals)
av_std = np.std(av_intervals)

# Respiration: Compute standard deviation as HRV proxy
resp_std = np.std(respiration)

# Combine features
features = np.array([rr_mean, rr_std, av_mean, av_std, resp_std])
print("Extracted features:", features)
```

**What to Do**:
- Create a new code cell and paste the code above.
- Run the cell after Step 4. It uses `clean_fECG`, `pwd_envelope`, and `respiration` from previous steps.
- **Expected Output**: Array of features, e.g., `[rr_mean, rr_std, av_mean, av_std, resp_std]`.
- **Tip**: If peak detection fails, adjust the `distance` parameter in `find_peaks` based on the data’s heart rate (e.g., 120–150 bpm for fetuses).

#### Step 6: Multimodal Fusion for FHR Estimation
**What**: Combine fECG and PWD to estimate fetal heart rate (FHR).
**Why**: Fusing modalities improves FHR accuracy, leveraging NInFEA’s synchronized data.
**How**:
1. Use a simple averaging approach to combine fECG and PWD heart rates (as a baseline; Kalman filtering or CNN can be added later).
2. Run the following code to compute FHR.

```python
# Step 6: Multimodal FHR estimation
# Explanation:
# - Compute FHR from fECG (R-peaks) and PWD (envelope peaks).
# - Average the two estimates for fusion.
# - Convert to beats per minute (bpm).

# fECG FHR: From RR intervals
fhr_fecg = 60 / rr_mean  # bpm

# PWD FHR: From atrioventricular intervals
fhr_pwd = 60 / av_mean  # bpm

# Fused FHR: Average of fECG and PWD
fhr_fused = (fhr_fecg + fhr_pwd) / 2

print("fECG FHR:", fhr_fecg, "bpm")
print("PWD FHR:", fhr_pwd, "bpm")
print("Fused FHR:", fhr_fused, "bpm")
```

**What to Do**:
- Create a new code cell and paste the code above.
- Run the cell after Step 5. It uses `rr_mean` and `av_mean` from Step 5.
- **Expected Output**: FHR values, e.g., `Fused FHR: 140.5 bpm`.
- **Tip**: If FHR values seem off (e.g., <100 or >200 bpm), check peak detection in Step 5.

#### Step 7: Machine Learning for Classification
**What**: Train a simple machine learning model (e.g., SVM) to classify normal vs. abnormal fetal states.
**Why**: NInFEA contains healthy fetuses, so we’ll simulate abnormal data (e.g., arrhythmias) for training.
**How**:
1. Generate synthetic abnormal data by perturbing RR intervals.
2. Train an SVM classifier using features from Step 5.
3. Run the following code.

```python
# Step 7: Train SVM for classification
# Explanation:
# - Since NInFEA has healthy fetuses, we simulate abnormal data by perturbing RR intervals.
# - We train an SVM to classify normal vs. abnormal (e.g., arrhythmia).
# - Features: rr_mean, rr_std, av_mean, av_std, resp_std.

from sklearn.svm import SVC
import numpy as np

# Simulate data for 60 records (mimicking NInFEA’s size)
n_samples = 60
normal_features = np.tile(features, (n_samples//2, 1))  # Normal: repeat healthy features
abnormal_features = normal_features + np.random.normal(0, 0.1, normal_features.shape)  # Abnormal: add noise
X = np.vstack([normal_features, abnormal_features])
y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))  # Labels: 0=normal, 1=abnormal

# Train SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X, y)

# Predict on sample features
pred = svm.predict([features])
print("Prediction (0=normal, 1=abnormal):", pred)
```

**What to Do**:
- Create a new code cell and paste the code above.
- Run the cell after Step 5. It uses `features` from Step 5.
- **Expected Output**: Prediction, e.g., `Prediction (0=normal, 1=abnormal): [0]`.
- **Tip**: For real data, we’d need labeled abnormal data. This is a simulation due to NInFEA’s healthy fetus limitation.

#### Step 8: Explainable AI with SHAP
**What**: Use SHAP to explain SVM predictions.
**Why**: Clinicians need interpretable results. SHAP shows which features (e.g., RR intervals) drive predictions.
**How**:
1. Apply SHAP to the SVM model.
2. Run the following code to generate explanations.

```python
# Step 8: Explain predictions with SHAP
# Explanation:
# - We use SHAP’s KernelExplainer to compute feature importance for the SVM.
# - Features are labeled for clarity (rr_mean, rr_std, etc.).
# - We plot a bar chart of SHAP values.

import shap
import matplotlib.pyplot as plt

# Define feature names
feature_names = ['RR Mean', 'RR Std', 'AV Mean', 'AV Std', 'Respiration Std']

# SHAP explainer
explainer = shap.KernelExplainer(svm.predict, X)
shap_values = explainer.shap_values([features])

# Plot SHAP values
shap.summary_plot(shap_values, features=[features], feature_names=feature_names, plot_type='bar')
plt.show()
```

**What to Do**:
- Create a new code cell and paste the code above.
- Run the cell after Step 7. It uses `svm`, `X`, and `features` from Step 7.
- **Expected Output**: A bar plot showing feature importance (e.g., RR Mean’s contribution to the prediction).
- **Tip**: If the plot doesn’t display, ensure `matplotlib` is installed (from Step 1) and re-run.

#### Step 9: Visualize Results
**What**: Plot fECG, PWD envelope, and FHR to visualize the pipeline’s output.
**Why**: Visualization helps verify signal quality and FHR estimates.
**How**:
1. Plot the clean fECG, PWD envelope, and FHR estimates.
2. Run the following code.

```python
# Step 9: Visualize results
# Explanation:
# - Plot clean fECG, PWD envelope, and FHR estimates.
# - Use matplotlib for plotting, with time axes for clarity.

import matplotlib.pyplot as plt

# Time axes
t_fecg = np.arange(len(clean_fECG)) / fs
t_pwd = np.arange(len(pwd_envelope)) / 60  # PWD at 60 Hz

# Plot
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t_fecg, clean_fECG)
plt.title('Clean fECG')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(t_pwd, pwd_envelope)
plt.title('PWD Envelope')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot([0, 1, 2], [fhr_fecg, fhr_pwd, fhr_fused], 'o-')
plt.xticks([0, 1, 2], ['fECG', 'PWD', 'Fused'])
plt.title('FHR Estimates')
plt.ylabel('FHR (bpm)')

plt.tight_lawet()
plt.show()
```

**What to Do**:
- Create a new code cell and paste the code above.
- Run the cell after Step 6. It uses `clean_fECG`, `pwd_envelope`, `fhr_fecg`, `fhr_pwd`, and `fhr_fused`.
- **Expected Output**: Three subplots showing fECG, PWD envelope, and FHR estimates.
- **Tip**: Adjust `figsize` if the plot is too small.

#### Step 10: Save and Share Results
**What**: Save the processed data and plots to Google Drive for persistence.
**Why**: Colab’s storage is temporary, so we save results to Google Drive.
**How**:
1. Mount Google Drive and save the features and plot.
2. Run the following code.

```python
# Step 10: Save results to Google Drive
# Explanation:
# - Mount Google Drive to save files.
# - Save features as a .csv and plot as a .png.

from google.colab import drive
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# Save features
features_df = pd.DataFrame([features], columns=['RR Mean', 'RR Std', 'AV Mean', 'AV Std', 'Respiration Std'])
features_df.to_csv('/content/drive/My Drive/ninfea_features.csv', index=False)

# Save plot
plt.savefig('/content/drive/My Drive/ninfea_plot.png')
print("Results saved to Google Drive!")
```

**What to Do**:
- Create a new code cell and paste the code above.
- Run the cell after Step 9. Follow the prompt to authenticate Google Drive access (click the link, copy the code, paste it).
- **Expected Output**: Confirmation message and files saved in the Google Drive’s root folder.
- **Tip**: Check the Google Drive to ensure `ninfea_features.csv` and `ninfea_plot.png` are saved.

### Notes for Beginners
- **Running Cells**: Run each cell in order (Step 1, then Step 2, etc.). Colab executes cells sequentially, and later steps depend on earlier ones.
- **Errors**: If we get errors, check:
  - File paths in Step 3 (use `!ls` to verify).
  - Dependencies in Step 1 (re-run if needed).
  - Runtime connection (reconnect if disconnected).
- **NInFEA Limitations**: The dataset has healthy fetuses, so we simulated abnormal data. For real applications, we’d need labeled pathological data.
- **Extending the Pipeline**: To add Kalman filtering or CNN-LSTM, we can expand Step 6 or 7. Let me know if we want code for those!

### Full Workflow Summary
1. **Setup**: Install libraries and create a directory.
2. **Download**: Fetch and extract NInFEA dataset.
3. **Load/Preprocess**: Read .bin and .bmp files, preprocess signals.
4. **fECG Extraction**: Use ICA to isolate fECG.
5. **Feature Extraction**: Compute RR intervals, AV intervals, and HRV.
6. **FHR Estimation**: Fuse fECG and PWD for accurate FHR.
7. **Machine Learning**: Train SVM for classification (simulated data).
8. **Explainability**: Use SHAP to interpret predictions.
9. **Visualization**: Plot signals and FHR estimates.
10. **Save**: Store results in Google Drive.

This pipeline leverages NInFEA’s multimodal data to achieve the framework’s goals, with clear outputs at each step. If we need help with a specific step or want to extend the code (e.g., for CNN-LSTM or Kalman filtering), let me know!
