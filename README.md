# Lung Disease Classification with CNNs

This project implements a Convolutional Neural Network (CNN)-based system to classify lung diseases using chest X-rays. The project uses three state-of-the-art CNN architectures—ResNet50, DenseNet121, and MobileNetV3—to evaluate model performance across accuracy, mAP (mean Average Precision), and training time.

---

## **Project Overview**

Lung disease classification is an essential task in medical imaging. This project trains and evaluates CNNs to classify chest X-rays into two categories:
- **Normal**
- **Pneumonia**

The models are trained and evaluated on a publicly available chest X-ray dataset and achieve high accuracy while minimizing validation loss. 

---

## **Features**

- **Data Preparation**: 
  - Dataset is divided into training, validation, and testing sets.
  - Corrupted images are automatically detected and removed.
  
- **Deep Learning Models**:
  - ResNet50
  - DenseNet121
  - MobileNetV3

- **Evaluation Metrics**:
  - Accuracy
  - Mean Average Precision (mAP)
  - Training Time

- **Data Visualization**:
  - Model accuracy and loss curves.
  - Confusion matrices for classification evaluation.

---

## **Dataset**

The dataset used in this project is derived from a publicly available chest X-ray dataset. It contains labeled images categorized as:
- **Normal** (No Findings)
- **Pneumonia**

### Dataset Preparation:
1. Filtered images into `Normal` and `Pneumonia` categories based on metadata labels.
2. Organized into folders:
   - `train`
   - `validation`
   - `test`
3. Removed corrupted images to ensure data integrity.

---

## **Models and Training**

### **CNN Architectures**
1. **ResNet50**:
   - A residual network optimized for deeper layers.
   - Achieved ~99.94% accuracy on the test set.

2. **DenseNet121**:
   - A densely connected network for efficient feature reuse.
   - Achieved perfect accuracy of 100% on the test set.

3. **MobileNetV3**:
   - A lightweight CNN optimized for mobile devices.
   - Achieved ~97.91% accuracy on the test set.

### **Training Process**
- All models trained for **50 epochs** or until early stopping was triggered.
- Optimized using the **Adam optimizer** with a learning rate of 0.0001.
- Validation loss monitored to save the best-performing weights.

### **Evaluation**
- Accuracy, mAP, and training time were recorded for each model.
- Confusion matrices were plotted for performance visualization.

---

## **Performance Summary**

| Model         | Test Accuracy | mAP    | Training Time  |
|---------------|---------------|--------|----------------|
| ResNet50      | 99.94%        | 0.4997 | 269 seconds    |
| DenseNet121   | 100.00%       | 0.7346 | 247 seconds    |
| MobileNetV3   | 97.91%        | 0.5018 | 675 seconds    |

### **Conclusion**
- **DenseNet121** performed the best with perfect accuracy and the highest mAP, making it the most suitable model for this classification task.
- **ResNet50** also performed exceptionally well and could be considered for faster training.
- **MobileNetV3**, while slightly lower in accuracy, offers a lightweight alternative for resource-constrained environments.

---

## **Setup and Usage**

### **Prerequisites**
- Python 3.7+
- TensorFlow 2.x
- Keras
- Jupyter Notebook
- Required Python libraries listed in `requirements.txt`

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Lung-Disease-Classification.git
   cd Lung-Disease-Classification
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```

---

## **File Structure**

- **`FinalLungProjAi.ipynb`**: Jupyter notebook containing the full implementation.
- **`README.md`**: Project documentation.
- **`requirements.txt`**: List of required Python libraries.
- **`models/`**: Folder for saving trained model weights.
- **`data/`**: Folder containing the dataset (train, validation, test splits).

---

## **How to Test the Models**

1. Use the `test_single_image` function to test a single image against the trained models.
2. Place an image in the specified directory and execute the function.
3. The models will classify the image and return predictions.

---

## **References**

- Dataset: [Chest X-ray Dataset from NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- Model Architectures:
  - [ResNet50](https://arxiv.org/abs/1512.03385)
  - [DenseNet121](https://arxiv.org/abs/1608.06993)
  - [MobileNetV3](https://arxiv.org/abs/1905.02244)
- TensorFlow Documentation: [TensorFlow Guide](https://www.tensorflow.org/guide)
- Keras Documentation: [Keras API](https://keras.io/api/)
- Tutorials and Articles:
  - Deep Learning for Image Classification (Video).
  - Medical Imaging with CNNs (Article).

---

## **License**
This project is licensed under the MIT License - see the `LICENSE` file for details.
