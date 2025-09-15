import requests

body = {
    "CreditScore": 650,
    "Age": 45,
    "Tenure": 6,
    "Balance": 50000,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 70000,
    "Geography_Germany": 1,
    "Geography_Spain": 0,
    "Gender_Male": 1
}

response = requests.post(
    url='http://127.0.0.1:8000/score',
    json=body
)

print(response.json())

