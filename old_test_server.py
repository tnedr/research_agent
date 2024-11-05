import requests
from requests.exceptions import RequestException

# BASE_URL = "http://localhost:10000"
# BASE_URL = "http://127.0.0.1:10000"
BASE_URL = 'https://research-agent2.onrender.com'


if False:
    response = requests.post(
            BASE_URL,
            json={
                "query": "what is nicotinamide riboside?",
            }
    )
    response = response.json()
    print(response)


def test_web_service(query):
    payload = {"query": query}
    try:
        response = requests.post(BASE_URL, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return response.json()
    except RequestException as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    query = "what is nicotinamide riboside?"
    # response = test_web_service(query)
    # if response is not None:
    #     print(response)