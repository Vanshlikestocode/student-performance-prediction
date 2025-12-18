import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['Hours_Studied', 'Attendance']]
y = data['Score']

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
hours = int(input("Enter hours studied: "))
attendance = int(input("Enter attendance percentage: "))

# Predict
input_data = pd.DataFrame([[hours, attendance]],
                          columns=['Hours_Studied', 'Attendance'])

prediction = model.predict(input_data)

print(f"Predicted Score: {prediction[0]:.2f}")

# Visualization
plt.scatter(data['Hours_Studied'], y)
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Hours Studied vs Score")
plt.show()
