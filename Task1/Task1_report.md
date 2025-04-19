# 📜 Breast Cancer Prediction Using Neural Networks

## 📄 Introduction
This project aims to predict the likelihood of breast cancer based on various features of tumor cells. The dataset used for this project contains information about the characteristics of cell nuclei present in breast cancer biopsies. Using machine learning techniques, we will analyze the data to classify whether a tumor is **malignant** or **benign**.

The goal of this project is to build a classification model to predict the diagnosis based on features like radius, texture, smoothness, compactness, etc.

---

## 🧑‍💻 Dataset Overview
The dataset contains several features describing the characteristics of the cell nuclei, including:

- **ID**: Unique identifier for each patient
- **Diagnosis**: Diagnosis of breast cancer (`M` = Malignant, `B` = Benign)
- **Radius**: Mean distance from center to points on the perimeter
- **Texture**: Standard deviation of gray-scale values
- **Perimeter**: Tumor perimeter
- **Area**: Tumor area
- **Smoothness**: Local variation in radius lengths
- **Compactness**: `(Perimeter^2 / Area) - 1.0`
- **Concavity**: Severity of concave portions of the contour
- **Concave Points**: Number of concave portions of the contour
- **Symmetry**: Tumor symmetry
- **Fractal Dimension**: Coastline approximation - 1

---

## 🗂️ Files in this Repository

- `Breast_Cancer_Prediction.ipynb`: Jupyter notebook with the full analysis, including preprocessing, EDA, and model training.
- `breast_cancer.csv`: Dataset used for analysis.
- `requirements.txt`: Python dependencies required to run the notebook.
- `model.pkl`: *(Optional)* Serialized model file for future predictions.

---

## 🧑‍💻 Techniques Used

### 1. Data Preprocessing
- Handled missing values (if any)
- Encoded categorical variables (e.g., Diagnosis)
- Feature scaling using **StandardScaler** for normalized input

### 2. Exploratory Data Analysis (EDA)
- Summary statistics
- Correlation heatmaps
- Distribution plots (histograms, boxplots)
- Pair plots to understand feature relationships

---

## 📈 Visualizations Used
- **Feature Distributions**: Plots for features like radius, area, and smoothness
- **Correlation Matrix**: Heatmap to visualize feature correlations
- **Pair Plots**: Relationship visualization between selected features and diagnosis

---

## 🪠 Neural Network Model

```python
model_ann = Sequential([
    InputLayer(shape=(30,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Epochs**: 50
- **Train-Test Split**: 80%-20% with 20% validation during training

### Evaluation:
- Accuracy and Loss plots for both training and validation
- Confusion matrix (heatmap)
- Classification report (precision, recall, f1-score)

---

## 🔍 Insights and Findings
- **Malignant vs. Benign**: Features like radius, area, and smoothness are significantly different between the two classes.
- **Feature Importance**: Feature correlation and model-based interpretation show radius and texture are among the most influential features.
- **Model Performance**: The ANN performed well with high accuracy and recall, proving effective for medical diagnosis tasks.

---

## 🛠️ Libraries Used
- `numpy`, `pandas`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Preprocessing and metrics
- `tensorflow.keras`: ANN implementation

---

## 📅 How to Run
```bash
pip install -r requirements.txt
```
Then open the Jupyter notebook:
```bash
jupyter notebook Breast_Cancer_Prediction.ipynb
```

---

## 🎓 License
This project is open source and free to use for educational and research purposes.
