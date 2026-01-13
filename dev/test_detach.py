import requests

response = requests.post(
    "http://localhost:26006/detach_physical_memory",
    json={},
)
