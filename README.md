# Heart_Disease_Detection_ML_Project


## 📌 Project Overview
This project is a **Machine Learning model** that predicts whether a person has heart disease based on medical data. The model is trained using the **Logistic Regression** algorithm and evaluates patient health metrics to provide predictions.

## 📂 Dataset Information
The dataset used for this project contains multiple medical parameters such as:
- **Age**
- **Sex**
- **Chest Pain Type**
- **Resting Blood Pressure**
- **Cholesterol Level**
- **Fasting Blood Sugar**
- **Resting ECG Results**
- **Maximum Heart Rate Achieved**
- **Exercise-Induced Angina**
- **Oldpeak (ST Depression)**
- **Slope of the Peak Exercise ST Segment**
- **Number of Major Vessels Colored by Fluoroscopy**
- **Thalassemia**
- **Target Variable (0 = No Disease, 1 = Disease Present)**

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## ⚙️ Project Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2️⃣ Install Dependencies
Make sure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook
```
Then open **Heart Disease Prediction.ipynb** and run all the cells.

## 📊 Model Training & Prediction
### Train the Model
```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

### Make a Prediction
```python
input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)
print("Prediction:", prediction)
```
**Output:**
- `1` → Patient has heart disease
- `0` → Patient does not have heart disease

## 📈 Model Performance
The model was evaluated using accuracy metrics:
```python
accuracy_score(Y_test, model.predict(X_test))
```
✅ **Training Accuracy:** XX%  
✅ **Testing Accuracy:** XX%

## 🚀 Future Improvements
- Use other ML models like **Random Forest, SVM, Neural Networks**.
- Improve accuracy with **feature selection & hyperparameter tuning**.
- Create a **Web App using Flask or Streamlit** for real-world usage.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests for improvements!

