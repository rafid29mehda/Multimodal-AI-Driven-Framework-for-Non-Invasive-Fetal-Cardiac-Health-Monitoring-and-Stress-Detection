# Multimodal-AI-Driven-Framework-for-Non-Invasive-Fetal-Cardiac-Health-Monitoring-and-Stress-Detection

The **NInFEA dataset** (Non-Invasive Multimodal Foetal ECG-Doppler Dataset for Antenatal Cardiology Research) provides a rich foundation for implementing the proposed **Multimodal AI-Driven Framework for Non-Invasive Fetal Cardiac Health Monitoring and Stress Detection**. By incorporating the specific characteristics of the NInFEA dataset, we can refine the methodology to leverage its unique features, such as simultaneous 27-channel electrophysiological recordings, fetal pulsed-wave Doppler (PWD) signals, maternal respiration signals, and clinical annotations. Below, I analyze the dataset and integrate it into the previously outlined framework, ensuring alignment with its structure and capabilities while enhancing the depth and impact of the work.

---

### Analysis of the NInFEA Dataset
The NInFEA dataset, published on November 12, 2020, by Pani et al., is a pioneering open-access resource designed for antenatal cardiology research. Its key features are:

1. **Data Composition**:
   - **Subjects**: 60 entries from 39 pregnant women, recorded between the 21st and 27th weeks of gestation, ensuring a focus on early pregnancy—a critical period for non-invasive monitoring.
   - **Signals**:
     - **Electrophysiological Data**: 27 channels (2048 Hz, 22-bit resolution) acquired via the TMSi Porti7 system:
       - Channels 1–24: Unipolar channels from maternal abdomen and back for fECG.
       - Channels 25–27: Differential channels from maternal thorax for maternal ECG.
       - Channel 32: Maternal respiration signal (piezo-resistive belt).
       - Channel 33: Internal saw-tooth signal.
       - Channel 34: Trigger signal for synchronization.
     - **PWD Signals**: Recorded via a Philips iE33 Ultrasound Machine (five-chamber apical window, 1680x1050 pixels, 60 Hz, 75 mm/s sweep speed), stored as .bmp images.
     - **Signal Length**: Varies from 7.5 s to 119.8 s (average 30.6 s ± 20.6 s).
   - **Clinical Annotations**: Provided by expert clinicians, including fetal echocardiography reports confirming healthy fetuses, serving as ground truth for validation.
   - **Metadata**: Includes fetal position from B-mode echography, aiding in contextual analysis.

2. **Unique Features**:
   - **Multimodality**: Combines fECG, PWD, and maternal respiration, enabling comprehensive fetal health analysis.
   - **Early Pregnancy Focus**: Captures data in the 21st–27th week, ideal for studying early cardiac development.
   - **High-Resolution Electrophysiology**: 27 channels with high sampling rate (2048 Hz) and spatial redundancy, supporting advanced signal processing (e.g., ICA, QRD-RLS).
   - **Synchronization**: Post-processed synchronization between fECG and PWD ensures temporal alignment, critical for multimodal fusion.
   - **Matlab Tools**: Includes a binary file reader, PWD envelope extraction code, and a graphical user interface for signal visualization, facilitating preprocessing and analysis.

3. **Limitations**:
   - **Healthy Fetuses**: All signals are from cardiologically healthy fetuses, limiting direct applicability to anomaly detection without augmentation or transfer learning.
   - **Variable Signal Length**: Short recordings (7.5 s minimum) may challenge temporal analyses like LSTM-based modeling.
   - **Synchronization**: Lack of direct hardware synchronization between Porti7 and iE33 requires robust post-processing to ensure alignment.
   - **Limited Sample Size**: 60 entries from 39 subjects may necessitate data augmentation for robust machine learning.

4. **Relevance to Framework**:
   - The dataset’s multimodal nature (fECG, PWD, respiration) directly supports the proposed framework’s focus on multimodal fusion.
   - Clinical annotations and echocardiography reports provide ground truth for validating FHR estimation, arrhythmia detection, and stress analysis.
   - Matlab tools align with the proposed signal processing pipeline (e.g., ICA, wavelet denoising, PWD envelope extraction).
   - Early pregnancy data (21st–27th weeks) aligns with the framework’s emphasis on early detection.

---

### Refined Framework Incorporating NInFEA Dataset
The proposed framework is tailored to leverage the NInFEA dataset’s strengths, address its limitations, and deliver impactful contributions. Below, I integrate the dataset into the five-module methodology, specifying how each module utilizes NInFEA’s data and tools.

#### 1. Fetal ECG Extraction and Denoising
**Objective**: Extract clean fECG from the 27-channel electrophysiological data.
- **NInFEA Integration**:
  - **Input Data**: Use the 27-channel electrophysiological signals (channels 1–24 for fECG, 25–27 for maternal ECG) from the .bin files.
  - **Preprocessing**:
    - Leverage the provided Matlab binary file reader to load .bin files into a Matlab variable for processing.
    - Apply **Independent Component Analysis (ICA)** to separate fECG from maternal ECG and noise, exploiting the high spatial redundancy of the 27 channels (Sulas et al., 2019).
    - Use **QRD-RLS adaptive filtering** (Sulas et al., 2019) with maternal ECG (channels 25–27) as a reference to refine fECG extraction.
    - Enhance SNR with **wavelet-based denoising** (Baldazzi et al., 2020), as validated in the NInFEA references, using the provided Matlab tools.
  - **Validation**:
    - Use PWD signals (.bmp files) and the provided Matlab PWD envelope extraction code to derive fetal heart activity as ground truth.
    - Compare extracted fECG with PWD-derived heartbeats to assess SNR improvement.
- **Adaptation to Dataset**:
  - Account for variable signal lengths (7.5–119.8 s) by standardizing analysis windows (e.g., 10 s segments) to ensure consistency.
  - Use metadata (fetal position from B-mode echography) to adjust electrode weighting in ICA, as fetal position affects signal quality.

**Output**: Clean fECG signals with high SNR, ready for feature extraction.

#### 2. Multimodal Feature Extraction
**Objective**: Extract comprehensive features from fECG, PWD, and maternal respiration for cardiac and stress analysis.
- **NInFEA Integration**:
  - **fECG Features**:
    - From clean fECG (channels 1–24), extract RR intervals, P-wave morphology, QRS complex characteristics, and HRV using Matlab-based signal processing.
    - Use the provided graphical user interface to visualize fECG and ensure accurate feature extraction.
  - **PWD Features**:
    - Process .bmp PWD images using the provided Matlab PWD envelope extraction code to compute atrioventricular time intervals, peak velocities, and envelope changes (Sulas et al., 2018).
    - Validate features against clinical annotations of fetal heart activity.
  - **Maternal Respiration Features**:
    - Extract respiration patterns from channel 32 (piezo-resistive belt) to capture maternal-fetal interactions, such as respiration-induced HRV changes.
  - **Synchronization**:
    - Use the trigger signal (channel 34) and post-processing synchronization techniques to align fECG, PWD, and respiration signals temporally.
    - Leverage metadata (e.g., fetal position) to contextualize feature variations.
- **Adaptation to Dataset**:
  - Address short signal lengths by focusing on high-quality segments (e.g., 10–30 s) and using data augmentation (e.g., synthetic signal generation) for shorter records.
  - Normalize features across subjects to account for inter-subject variability (e.g., maternal physiology, fetal position).

**Output**: A synchronized, multimodal feature set capturing cardiac (RR intervals, P-wave morphology) and stress-related (HRV, Doppler envelope changes) patterns.

#### 3. Multimodal Sensor Fusion and FHR Estimation
**Objective**: Combine fECG and PWD for accurate FHR estimation.
- **NInFEA Integration**:
  - **Input Data**: Clean fECG (from module 1), PWD signals (.bmp files), and extracted features.
  - **Fusion Techniques**:
    - Apply **Kalman filtering** to fuse fECG-derived heartbeats (from channels 1–24) and PWD-derived heartbeats (from envelope extraction), using the trigger signal (channel 34) for synchronization.
    - Alternatively, train a **multimodal CNN** to integrate fECG waveforms and PWD envelopes, learning optimal weights for FHR estimation.
  - **Validation**:
    - Compare fused FHR estimates with single-modality estimates (fECG-only, PWD-only) using clinical annotations as ground truth.
    - Evaluate accuracy metrics (e.g., mean absolute error) across the 60 entries, focusing on early pregnancy (21st–27th weeks).
- **Adaptation to Dataset**:
  - Use the high-resolution PWD (60 Hz, 75 mm/s sweep speed) to enhance FHR precision, compensating for fECG noise in short recordings.
  - Account for healthy fetus data by simulating arrhythmic or anomalous FHR patterns via synthetic augmentation for robustness testing.

**Output**: Reliable FHR estimates with reduced errors, validated against NInFEA’s clinical annotations.

#### 4. Machine Learning for Arrhythmia, Anomaly, and Stress Detection
**Objective**: Develop a multi-task model to detect arrhythmias, anomalies, and stress.
- **NInFEA Integration**:
  - **Input Data**: Multimodal features from module 2, clinical annotations, and echocardiography reports.
  - **Model Architecture**:
    - Train a **hybrid CNN-LSTM network**:
      - CNN layers process fECG waveforms (channels 1–24) and PWD envelopes (.bmp files) for spatial feature extraction.
      - LSTM layers model temporal dependencies in RR intervals, HRV, and Doppler changes.
    - **Tasks**:
      - **Arrhythmia Detection**: Classify normal vs. arrhythmic beats (e.g., premature atrial contractions) using fECG and PWD features. Since NInFEA includes healthy fetuses, use synthetic data or transfer learning from other datasets (e.g., PhysioNet’s fetal scalp ECG) to simulate arrhythmias.
      - **Anomaly Detection**: Identify potential structural or conduction anomalies. Augment NInFEA with simulated anomalies (e.g., altered QRS morphology) or external datasets to train the model, validated against echocardiography reports.
      - **Stress Detection**: Detect hypoxia-related stress using HRV (fECG), Doppler envelope changes (PWD), and maternal respiration (channel 32). Simulate stress patterns (e.g., HRV suppression) due to healthy fetus data.
  - **Training**:
    - Use cross-validation across the 60 entries to ensure generalizability.
    - Apply data augmentation (e.g., SMOTE, synthetic signal generation) to address limited sample size and healthy fetus limitation.
    - Incorporate early pregnancy focus (21st–27th weeks) to prioritize early detection.
  - **Validation**:
    - Validate arrhythmia and stress detection using PWD-derived ground truth and clinical annotations.
    - For anomaly detection, use echocardiography reports as a reference, supplemented by external datasets for pathological cases.
- **Adaptation to Dataset**:
  - Address the healthy fetus limitation by integrating external datasets or generating synthetic pathological signals (e.g., via OSET toolbox, Sameni, 2018).
  - Handle variable signal lengths by standardizing input windows (e.g., 10 s) for LSTM processing.

**Output**: A multi-task model for arrhythmia, anomaly, and stress detection, validated on NInFEA and augmented data.

#### 5. Explainable AI for Clinical Interpretability
**Objective**: Provide interpretable predictions to enhance clinician trust.
- **NInFEA Integration**:
  - **Input Data**: Model predictions, multimodal features, clinical annotations.
  - **Explainability**:
    - Apply **SHAP** to quantify feature contributions (e.g., RR intervals, PWD velocities) to predictions, using Matlab tools for visualization.
    - Use **LIME** to generate visual explanations of individual predictions, highlighting key fECG (channels 1–24) and PWD patterns.
    - Integrate explanations into the provided Matlab graphical user interface, displaying fECG, PWD envelopes, and prediction rationales.
  - **Validation**:
    - Engage clinicians from the Pediatric Cardiology and Congenital Heart Disease Unit (Brotzu Hospital) to review explanations, ensuring alignment with NInFEA’s clinical annotations.
    - Use metadata (e.g., fetal position) to contextualize explanations, improving clinical relevance.
- **Adaptation to Dataset**:
  - Leverage the high-resolution 27-channel data to provide granular feature importance (e.g., specific electrode contributions).
  - Use the provided GUI to create a clinician-friendly dashboard, integrating fECG, PWD, FHR estimates, and SHAP/LIME outputs.

**Output**: Interpretable predictions with a clinician-friendly dashboard, enhancing adoption.

---

### Impactful Contributions with NInFEA
By leveraging the NInFEA dataset, the framework delivers:
1. **Robust Non-Invasive Monitoring**: Uses NInFEA’s 27-channel fECG and PWD to develop reliable algorithms for fECG extraction and FHR estimation, validated on early pregnancy data (21st–27th weeks).
2. **Early Detection**: Adapts NInFEA’s healthy fetus data with synthetic augmentation to enable early detection of arrhythmias, anomalies, and stress, critical for antenatal care.
3. **Comprehensive Insights**: Combines fECG, PWD, and respiration signals to provide holistic fetal health assessments, supported by NInFEA’s multimodal design.
4. **Clinical Trust**: Integrates SHAP/LIME explanations into NInFEA’s Matlab GUI, creating a clinician-friendly tool validated by experts at Brotzu Hospital.
5. **Research Advancement**: Contributes to fetal cardiac physiology research by analyzing NInFEA’s high-resolution data, with open-source code and publications citing the dataset (Pani et al., 2020; Sulas et al., 2021).
6. **Scalability**: Adapts NInFEA’s processing tools for potential integration into wearable devices, enhancing clinical applicability.

---

### Addressing Dataset Limitations
1. **Healthy Fetuses**:
   - **Mitigation**: Augment NInFEA with synthetic pathological signals (e.g., via OSET toolbox) or external datasets (e.g., PhysioNet’s fetal scalp ECG) for arrhythmia and anomaly detection.
2. **Variable Signal Length**:
   - **Mitigation**: Standardize analysis windows (e.g., 10 s) and use data augmentation to extend short recordings.
3. **Limited Sample Size**:
   - **Mitigation**: Apply cross-validation and transfer learning to maximize the 60 entries’ utility.
4. **Synchronization**:
   - **Mitigation**: Use NInFEA’s trigger signal (channel 34) and post-processing techniques to ensure robust alignment.

---

### Deliverables with NInFEA
1. **Algorithms**: Open-source Matlab code for fECG denoising, multimodal fusion, and AI models, compatible with NInFEA’s tools.
2. **Dashboard**: Enhanced version of NInFEA’s Matlab GUI, integrating fECG, PWD, FHR estimates, and explainable AI outputs.
3. **Publications**: Peer-reviewed papers citing NInFEA (Pani et al., 2020; Sulas et al., 2021; Goldberger et al., 2000) and detailing the framework’s methodology and results.
4. **Guidelines**: Recommendations for using NInFEA-based tools in antenatal care, aligned with clinical protocols.

---

### Example Visualization
To illustrate the framework’s performance, I can generate a chart comparing the accuracy of FHR estimation across modalities (fECG-only, PWD-only, fused) using simulated metrics from NInFEA’s 60 entries. Here’s a sample chart:

```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["fECG-only", "PWD-only", "Fused (Kalman)", "Fused (CNN)"],
    "datasets": [{
      "label": "FHR Estimation Accuracy (%)",
      "data": [85, 88, 92, 94],
      "backgroundColor": ["#4e79a7", "#f28e2b", "#76b7b2", "#59a14f"],
      "borderColor": ["#4e79a7", "#f28e2b", "#76b7b2", "#59a14f"],
      "borderWidth": 1
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": { "display": true, "text": "Accuracy (%)" }
      },
      "x": {
        "title": { "display": true, "text": "Method" }
      }
    },
    "plugins": {
      "title": { "display": true, "text": "FHR Estimation Accuracy Using NInFEA Dataset" }
    }
  }
}
```

This chart visualizes hypothetical accuracy improvements from multimodal fusion, leveraging NInFEA’s data. If we’d like a different chart (e.g., model performance for arrhythmia detection), please specify!

---

### Conclusion
The NInFEA dataset’s multimodal, high-resolution data and accompanying Matlab tools enable the implementation of a robust, AI-driven framework for fetal cardiac health monitoring. By integrating NInFEA’s 27-channel fECG, PWD, and respiration signals, the framework achieves accurate FHR estimation, early detection of arrhythmias and stress, and interpretable predictions, all validated against clinical annotations. Addressing NInFEA’s limitations through augmentation and transfer learning ensures applicability to pathological cases, delivering transformative contributions to antenatal care and research.
