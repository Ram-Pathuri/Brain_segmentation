# Brain Segmentation & Abnormality Detection — Project README

## Project Overview
This project implements an open-source AI framework for **automated 3D brain segmentation**, **abnormality (tumour) detection**, and **statistical analysis** of neuroimaging data (MRI). The system is built to run in cloud environments (AWS S3 + SageMaker) and produces segmentation maps, uncertainty quantification, and statistical comparisons between cohorts.

## Key Components (Architecture)
- **Data Sources**
  - IXI Dataset (T1-weighted MRI) — primary segmentation dataset.
  - BR35H dataset — tumor classification dataset.
  - Raw data stored in **AWS S3**.

- **Preprocessing**
  - Intensity normalization (standardization).
  - Resampling to consistent voxel spacing.
  - Data augmentation (affine transforms, intensity shifts, flips, rotations).
  - Conversion to 3D numpy arrays: **160 × 192 × 224** normalized to [0,1].
  - Label mapping with 24 anatomical classes.

- **Segmentation Model**
  - **3D U-Net** with **attention mechanisms** for improved localization.
  - Loss: **Dice Loss**.
  - Optimizer: **Adam** with LR scheduling.
  - Batch size: **4**.
  - Training performed on **AWS SageMaker** (e.g., `ml.m5.12xlarge`).

- **Tumor Detection Model**
  - **EfficientNetB0** (2D classification pipeline).
  - Input shape: **224 × 224 × 3**.
  - Loss: **Binary Cross Entropy**.
  - Optimizer: **Adam**, Batch size = 4.

- **Postprocessing**
  - Label smoothing / morphological operations as needed.
  - Volume calculations and per-region statistics.
  - Uncertainty quantification (e.g., Monte Carlo dropout or model ensembling).

- **Statistical Analysis**
  - Mann–Whitney U test used to compare segmented region volumes across cohorts.
  - Example result from dataset: `U = 20301.000, p-value = 1.000` (no significant difference in that analysis).

- **Evaluation**
  - Metrics: **Dice score**, confusion matrix, accuracy (reported in PDF: 51% segmentation accuracy; 20% tumor detection accuracy).
  - Train/Val/Test split: **576 volumes** split 7:1:2 → 403/58/115.

- **Deployment**
  - Models deployed as SageMaker endpoints.
  - An example sklearn pipeline implemented for converting raw inputs to predictions for easier deployment.



## How Data Flows (high-level)
1. Raw MRI and labels are uploaded to **S3**.
2. Preprocessing jobs (local or on SageMaker) read S3 data, normalize, resample, augment, and save processed numpy arrays back to S3.
3. Training pipelines read processed arrays and train the **3D U-Net** for segmentation and **EfficientNetB0** for tumor classification.
4. Trained models produce segmentation masks and classification outputs.
5. Postprocessing converts masks to region volumes; statistical scripts perform Mann–Whitney U tests and save reports.
6. Models are wrapped into SageMaker endpoints or an sklearn pipeline for deployment.



## Notes & Observations 
- Segmentation model used a 3D U-Net with attention and achieved **~51% accuracy** (Dice or accuracy unclear in report).
- Tumor detection used EfficientNetB0 and reported **~20% accuracy** — suggests either class imbalance, limited data, or training/config issues.
- Statistical testing (Mann–Whitney U) reported p-value **= 1.000**, indicating no detected difference for the compared cohorts in the provided analysis.
- Consider improving:
  - Data augmentation and balancing for tumor dataset.
  - Loss function tuning (e.g., combined Dice + BCE).
  - Validation of preprocessing pipeline (ensure consistent intensity scaling).
  - Uncertainty propagation and clear evaluation metrics (report Dice, IoU, sensitivity, specificity).

---

## Attached Architecture Diagram
![Brain Segmentation Architecture](https://github.com/Ram-Pathuri/Brain_segmentation/blob/main/brain_architecture.png)


---

## References & Links
- IXI Dataset: https://brain-development.org/ixi-dataset/
- Project repo (from PDF): https://github.com/Ram-Pathuri/Brain_segmentation

