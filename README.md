# Material Defect Detection (NEU-DET Dataset)

This project uses transfer learning with ResNet18 to classify six types of surface defects in steel, based on the NEU Surface Defect Database.

## Project Structure
```text
material-defect-detection/
├── data/                        # NEU-DET dataset (not included in repo)
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   └── validation/
│       ├── images/
│       └── annotations/
├── notebooks/                   # Jupyter notebook for training & evaluation
│   └── defect_detection.ipynb
├── models/                      # Saved PyTorch model weights
│   └── best_model.pth
├── classification_report.csv   # Per-class precision, recall, F1 score
├── confusion_matrix.csv        # Confusion matrix (validation set)
├── train.py                    # Script version of training pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```


## Dataset
- [NEU Surface Defect Database on Kaggle](https://www.kaggle.com/datasets/kaustubhkdishit/neu-surface-defect-database)
- 1800 grayscale images in 6 classes (300 per class)

## Approach
- Used pretrained ResNet18 from PyTorch's `torchvision.models`
- Froze layer1 and layer2 blocks for transfer learning
- Trained final layers for 5 epochs
- Achieved 99.7%+ accuracy on validation set

## Evaluation
- Confusion matrix & classification report included
- See `classification_report.csv` and `confusion_matrix.csv`

## Requirements
```bash
torch
torchvision
matplotlib
scikit-learn
pandas
jupyter
```

## Usage
To train the model:
```bash
python train.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook notebooks/defect_detection.ipynb
```