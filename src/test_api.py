import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "The stock market rallied today after the Fed announcement"}
)
print(response.json())