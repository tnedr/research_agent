import requests

print(
    requests.post(
        # "http://localhost:10000",
        "http://127.0.0.1:10000",
        json={
            "query": "what is nicotinamide riboside?",
        }
    ).json()
)