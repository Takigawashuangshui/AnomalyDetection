# Anomaly Detection System for MVTec AD
This is a system capable of anomaly detection for two distinct products from the MVTec Anomaly Detection dataset, screw and metal nut.
![alt text](https://github.com/Takigawashuangshui/AnomalyDetection/blob/main/pipeline_chart.png?raw=true)
The whole pipeline is implemented via pipeline.py.


## Methods 
The machine learning model is based on EfficientAD. https://arxiv.org/abs/2303.14535. The code is built based on https://github.com/nelson1425/EfficientAD.

By using Lightweight Student–Teacher + Autoencoder architecture for anomalies detection and Patch description networks (PDN) for feature extraction. The model enables a fast handling of anomalies with low error rate, making it a perfect choise for abnomaly detection in manufacturing industry.

### Challenges
1. Types and possible locations of defects are unknown --> Use Student–Teacher structure so that it perform well even trained only on normal images.

2. Industrial settings requires strict runtime limits --> Reduce computational cost by drastically reducing depth for feature extractor, performing down-sampling early, and using light-weight S-T structure.

3. Violations of logical constraints regarding the position, size, arrangement, etc. --> Use Autoencoder to detect logical anomalies.



## Results

![alt text](https://github.com/Takigawashuangshui/AnomalyDetection/blob/main/example.png?raw=true)

### Mean anomaly detection AU-ROC percentages:

| Product       | Model          | AU-ROC         |
|---------------|----------------|----------------|
| screw         | EfficientAD-S  | 96.8           |
| screw         | EfficientAD-M  | 97.4           |
| metal nut     | EfficientAD-S  | 98.7           |
| metal nut     | EfficientAD-M  | 99.3           |



### Computational efficiency: Latency

| Model         | GPU             | Latency      |
|---------------|-----------------|--------------|
| EfficientAD-S | Quadro RTX 6000 | 4.0 ms       |
| EfficientAD-M | Quadro RTX 6000 | 5.9 ms       |



## Setup

### Packages

```
Python==3.10
numpy==1.18.5
torch==1.13.0
torchvision==0.14.0
scikit-learn==1.2.2
tifffile==2021.7.30
tqdm==4.56.0
Pillow==7.0.0
scipy==1.7.1
tabulate==0.8.7
```

### Dataset

Download dataset (if you already have downloaded then set path to dataset (`--mvtec_ad_path`) when calling `efficientad.py`).

```
mkdir mvtec_anomaly_detection
cd mvtec_anomaly_detection
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz
cd ..
```


## Usage

Training and inference for screw and metal nut:

```
python mvtec_ad_training/efficientad_2objects.py --dataset mvtec_ad
```
Training with EfficientAD-M for two kinds of images:

```
python mvtec_ad_training/efficientad_2objects.py --model_size medium --weights models/teacher_medium.pth --dataset mvtec_ad
```

Evaluation with Mvtec evaluation code:

```
python mvtec_ad_evaluation/evaluate_experiment.py --anomaly_maps_dir './output/4/anomaly_maps/mvtec_ad/' --output_dir './output/4/metrics/mvtec_ad/' --evaluated_objects screw
```

Anomaly detection pipeline for one image:

```
python pipeline.py --sample_path './mvtec_anomaly_detection/metal_nut/test/scratch/000.png'
```
