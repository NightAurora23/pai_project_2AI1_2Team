Medical Cost Personal Dataset (Linear Regression)

---

 Team Member Details
Member 1 - Daman Basotra: 2503031240025  

Member 2 - Dhruv Kumar: 2503031240029  


---

 Problem Statement
The objective of this project is to predict medical insurance charges based on individual attributes such as age, gender, BMI, number of children, smoking habits, and region.  
The goal is to identify key factors affecting insurance costs and build a machine learning model to estimate charges accurately.

---

 Dataset Description
The dataset used in this project is taken from Kaggle Insurance Datset.

Feature    Description 

age        Age of the individual 
sex        Gender (male/female) 
bmi        Body Mass Index 
children   Number of dependents 
smoker     Smoking status (yes/no) 
region     Residential region 
charges    Medical insurance charges (target variable) 

The dataset is structured and contains no missing values.

---

 Data Preprocessing Steps
- Checked dataset for null/missing values
- Converted categorical variables into numerical format using encoding:
  - sex → 0/1
  - smoker → 0/1
  - region → label encoding
- Split dataset into training and testing sets
- Feature scaling applied (if required)

---

 Model Used and Training Details
- **Model Used:** Linear Regression  
- **Library:** Scikit-learn  
- The dataset was split into:
  - Training set (80%)
  - Testing set (20%)
- Model was trained using training data and predictions were made on test data

---

 Model Evaluation Results
- The model performance was evaluated using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

---

- Key Observations:
  - Smoking has a significant impact on insurance charges
  - Higher BMI leads to higher charges
  - Age also contributes to increased costs

---

 GitHub Collaboration Summary
- Repository created and managed on GitHub
- Code and dataset uploaded to the repository
- Version control maintained using commits
- README file added for documentation
- Project structured for easy understanding and execution

---

 Conclusion
This project successfully demonstrates how machine learning can be used to predict insurance charges.  
Linear Regression provides a good baseline model for this dataset.  
Further improvements can be made using advanced algorithms and hyperparameter tuning.
