## Machine Learning to support Clinical Decision Making in Patients with Acute Abdominal Pain

### Problem
- Patients with acute abdominal pain have to be assessed several times a day.
- It is often difficult to distinguish between patients with findings requiring urgent treatment from patients who do not require urgent treatment.
- The assessment must also be performed under difficult conditions (e.g., at night, under time pressure, in parallel).

### Methods and prerequisites
- The data contains missing values. Although missing values can be imputed but this could result in overfitting.
- Using ML methods that allow missing values in data.
- We tested 3 variants of random forest for this problem: h2o random forest, cforest and gradient boosting (xgboost).



