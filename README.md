🚗 Off-Road Semantic Segmentation using DeepLabV3+
📌 Project Overview

This project focuses on semantic segmentation of off-road environments for autonomous navigation and terrain understanding. 
The goal is to accurately classify each pixel in an image into predefined terrain categories such as trees, bushes, rocks, sky, and ground clutter.
We implemented a DeepLabV3+ model with a pretrained backbone and optimized it using advanced data augmentation strategies and a hybrid loss function.
Through iterative experimentation and systematic performance evaluation, we improved the Intersection over Union (IoU) score from an initial baseline
of [Baseline IoU] to a final score of [Final IoU], demonstrating strong segmentation performance and improved generalization capability.

🎯 Objectives

Perform pixel-level classification for 10 terrain classes
Improve IoU score through experimentation
Handle class imbalance effectively
Ensure model generalizes to unseen off-road scenes
Maintain computational efficiency

🧠 Model Architecture

🔹 Base Model
Architecture: DeepLabV3+
Backbone: Pretrained CNN backbone (ResNet / MobileNet)
Pretraining: ImageNet
🔹 Custom Modifications

Hybrid loss function:
Cross-Entropy Loss
Dice Loss
Class-weighted loss to handle imbalance
Strong data augmentation pipeline

🔹 Input Configuration
Image Resolution: 448 × 448
Number of Classes: 10
Optimizer: Adam
Learning Rate: 5e-5
Batch Size: 2
Epochs: 20



🚀 Key Improvement
Improved Mean IoU from [Baseline IoU] → [Final IoU]
Better boundary detection
Reduced misclassification in small-object classes
Improved performance on underrepresented classes

🧪 Training Strategy
Data Augmentation
Random Horizontal Flip
Color Jitter (Brightness, Contrast, Saturation)
Random Resizing
Normalization
Loss Function

We used a Hybrid Loss:

Total Loss = CrossEntropy + Dice Loss

This allowed:

Better class separation (CrossEntropy)
Improved overlap quality (Dice)


📈 Evaluation Metrics

The following metrics were used:
Mean Intersection over Union (IoU) (Primary Metric)
Dice Coefficient
Pixel Accuracy
IoU was used as the primary evaluation benchmark.

📂 Dataset Structure
Dataset/
├── train/
│   ├── Color_Images/
│   └── Segmentation/
├── val/
│   ├── Color_Images/
│   └── Segmentation/
└── test/
    ├── Color_Images/
    └── Segmentation/

Training outputs:
Model weights
Loss curves
IoU curves
Dice curves
Pixel accuracy curves
Evaluation logs

Output:
Mean IoU on Test Set: [Final IoU]

🏆 Key Highlights

Pretrained backbone for strong feature extraction
Hybrid loss for improved segmentation overlap
Class imbalance handled using weighted loss
Extensive metric tracking and visualization
Hackathon-ready modular codebase

📊Result and Performance
To systematically evaluate our improvements, we conducted four experimental runs, progressively enhancing the model with data augmentation and optimization strategies.

🔎 Observation 1 – Baseline Model
-DeepLabV3+ with pretrained backbone
-Standard preprocessing
-Cross-Entropy loss
-No augmentation
IoU Score: 0.2130
Analysis:
The baseline model achieved an IoU of 0.2130, indicating limited generalization and highlighting the domain gap challenge.

🔎 Observation 2 – Data Augmentation Applied
-Rotation, flipping, scaling
-Brightness and contrast adjustments
-Noise and blur augmentation
-IoU Score: 0.2870
Analysis:
Applying augmentation significantly improved robustness and generalization.

📈 Improvement from Baseline:
0.2130 → 0.2870
≈ 30.45% relative improvement

🔎 Observation 3 – Augmentation + Optimizating
-Hybrid loss (Cross-Entropy + IoU-based loss)
-Lightweight backbone
-Hyperparameter tuning
IoU Score: 0.2930
Analysis:
Further optimization improved model stability and inference efficiency.

📈 Improvement from Baseline:
0.2130 → 0.2930
≈ 33.18% relative improvement

🔎 Observation 4 – Final Fine-Tuned Model
-Refined hyperparameter tuning
-Improved training stability
-Better convergence strategy
IoU Score: 0.2941
Analysis:
The final configuration achieved the highest IoU of 0.2941, representing the best overall performance. While the increase over Observation 3 is incremental, it reflects improved model refinement and consistent convergence behavior.

📈 Overall Improvement from Baseline:
0.2130 → 0.2941
≈ 33.68% relative improvement

🔍 Key Insight
-Major improvement came from data augmentation.
-Optimization strategies provided incremental but important performance gains.
-Fine-tuning improved model stability and convergence consistency.

📌 Future Improvements

Add attention-based refinement module
Experiment with larger backbone (ResNet101)
Apply Test-Time Augmentation
Use focal loss for rare classes
Try transformer-based segmentation models
👨‍💻 Team
Developed for Off-Road Semantic Segmentation Hackathon.
