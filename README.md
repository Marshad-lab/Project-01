# Project-01
Computer vision based on Convolutional Neural Network

# 🐾 Animal Image Classification using ResNet18

This project implements an **animal image classification system** using
**Convolutional Neural Networks (CNNs)** and **transfer learning with ResNet18**.
The model classifies images into multiple animal categories with high accuracy.

---

Project Highlights
------------------

- Custom image loading using Pandas DataFrame
- Image validation and preprocessing
- Data augmentation for better generalization
- Transfer learning using pretrained **ResNet18**
- Hyperparameter tuning:
  - Learning rate
  - Dropout rate
  - Data augmentation
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

---

Model Architecture
-------------------

- Backbone: **ResNet18 (ImageNet pretrained)**
- Custom fully connected head
- Frozen backbone during transfer learning
- Optimizer: Adam
- Loss: CrossEntropyLoss

---

Dataset
------------
- 9 animal classes:
  - bee, butterfly, cat, lion, owl,
    panda, parrot, penguin, zebra
- 60 images per class
- Images resized to **224×224**

---

Technologies Used
-----------------

- Python
- PyTorch
- Torchvision
- OpenCV (PIL)
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn

---

 How to Run
 ----------

1. Clone repository

``git clone https://github.com/your-username/animal-image-classification-resnet.git
cd animal-image-classification-resnet``

2. Install dependencies
   
pip install -r requirements.txt

3. Run notebook

Open: Untitled0.ipynb

//Update dataset path and run all cells//

Evaluation:

  Normalized confusion matrix
  Class-wise precision, recall, and F1-score
  Excellent generalization with data augmentation

