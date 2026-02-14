# Fashion-MNIST Classification using ANN, CNN & Transfer Learning

Comparative study of Artificial Neural Network (ANN), Convolutional Neural Network (CNN), and Transfer Learning approaches on the Fashion-MNIST dataset using PyTorch.  
This project evaluates performance improvements achieved through spatial feature extraction and pretrained model weights.

---

## 📌 Project Objective

To compare the performance of:

- Fully Connected Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Transfer Learning using pretrained model weights

The goal is to analyze how architecture choice impacts classification accuracy and generalization.

---

## 📂 Dataset

**Fashion-MNIST**
- 70,000 grayscale images
- Image size: 28 × 28
- 10 clothing categories
- 60,000 training samples
- 10,000 test samples

---

## 🧠 Models Implemented

### 1️⃣ Artificial Neural Network (ANN)
- Fully connected layers
- Flattened image input
- ReLU activation
- CrossEntropy loss

**Test Accuracy: 83%**  
**Macro F1-Score: 0.83**

Limitations:
- Does not preserve spatial information
- Lower recall for visually similar classes

---

### 2️⃣ Convolutional Neural Network (CNN)
- Convolution layers
- MaxPooling
- Fully connected output layer
- ReLU activation

**Test Accuracy: 86%**  
**Macro F1-Score: 0.87**

Improvements over ANN:
- Learns spatial features
- Better class separation
- Improved recall for difficult classes

---

### 3️⃣ Transfer Learning (Pretrained Model)
- Pretrained CNN backbone (e.g., VGG16)
- Frozen feature extractor
- Custom classifier layer
- Fine-tuned final layers

**Test Accuracy: 90%**  
**Macro F1-Score: 0.90**

Improvements over CNN:
- Uses pretrained ImageNet features
- Faster convergence
- Better generalization
- Stronger performance across all classes

---

## 📊 Performance Comparison

| Model | Accuracy | Macro F1 | Improvement |
|--------|----------|----------|-------------|
| ANN | 83% | 0.83 | Baseline |
| CNN | 86% | 0.87 | +3% |
| Transfer Learning | 90% | 0.90 | +7% over ANN |

---

## 🔍 Key Observations

- ANN struggles with spatial relationships in image data.
- CNN significantly improves performance by extracting local patterns.
- Transfer learning achieves the highest accuracy by leveraging pretrained visual features.
- The largest improvements were observed in visually similar clothing categories.

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Train vs Test Accuracy Comparison

---

## 🧪 Conclusion

The experiment demonstrates a clear performance progression:

ANN (83%) → CNN (86%) → Transfer Learning (90%)

The results confirm that:
- Convolutional architectures are essential for image classification.
- Pretrained model weights significantly enhance performance.
- Transfer learning provides the best generalization capability.

---

## ⚙️ Tech Stack

- Python
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib

---

## 📬 Author

Your Name  
GitHub: Your Profile Link

