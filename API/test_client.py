# test_client.py
import requests

url = "http://127.0.0.1:8000/predict"
file = {'file': open("example.jpg", 'rb')}

response = requests.post(url, files=file)

print("Response JSON:")
print(response.json())