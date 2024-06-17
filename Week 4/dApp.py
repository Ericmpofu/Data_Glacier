
import pandas as pd

# Example toy data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)
print(df)


from sklearn.linear_model import LinearRegression
import joblib

# Example model training
X = df[['Age']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'linear_regression_model.joblib')
