import requests

url = "http://localhost:5000/api"

# restore deleted experiment
endpoint = "name/of/endpoint"
payload = {"key": "value"}

# request post
resp = requests.post(f"{url}/{endpoint}", json=payload, timeout=10)

print("status:", resp.status_code)
print("response:", resp.text)
resp.raise_for_status()
