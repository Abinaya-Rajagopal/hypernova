"""
granite_rag.py
------------------------------------
Reusable IBM Watson Granite + RAG client for contextual AI reasoning.
Can be imported into any script (e.g., touch_to_sense.py, painfeedback.py)
without modifying existing code.
------------------------------------
Usage:
    from granite_rag import GraniteRAGClient
    granite = GraniteRAGClient(API_KEY, URL)
    response = granite.query("Explain high phantom limb pain.")
"""

import requests
import json

class GraniteRAGClient:
    def __init__(self, api_key: str, url: str, assistant_id: str = None):
        """
        Initialize Granite Watson Assistant connector.

        :param api_key: Your IBM Cloud API key
        :param url: Watson Assistant service URL
        :param assistant_id: Optional Assistant ID if you use a specific workspace
        """
        self.api_key = api_key
        self.url = url.rstrip("/")
        self.assistant_id = assistant_id
        self.headers = {"Content-Type": "application/json"}

    def query(self, user_message: str, context: dict = None) -> str:
        """
        Sends a message to the IBM Watson Assistant (Granite model)
        and retrieves the model’s response.
        """
        try:
            # Build endpoint for Watson API
            if self.assistant_id:
                endpoint = (
                    f"{self.url}/v2/assistants/{self.assistant_id}/message?version=2021-06-14"
                )
            else:
                endpoint = f"{self.url}/v1/workspaces"

            payload = {
                "input": {"text": user_message},
                "context": context or {}
            }

            # Send request
            response = requests.post(
                endpoint,
                headers=self.headers,
                auth=("apikey", self.api_key),
                json=payload,
                timeout=10
            )

            # Parse result
            if response.status_code == 200:
                data = response.json()
                # Extract Watson’s reply text
                if "output" in data:
                    if "generic" in data["output"]:
                        return data["output"]["generic"][0].get("text", "")
                    elif "text" in data["output"]:
                        return data["output"]["text"][0]
                return "Granite returned no response text."
            else:
                return f"Error {response.status_code}: {response.text[:200]}"

        except Exception as e:
            return f"[Granite Error] {str(e)}"


# ==============================================================
# Example usage (only runs if this file is executed directly)
# ==============================================================

if __name__ == "__main__":
    # Replace these with your real credentials
    API_KEY = "b1aa235f207d3b4f520491ffde670885"
    URL = "https://api.au-syd.assistant.watson.cloud.ibm.com/instances/157f16b6-3106-467b-975b-da2490c562d5"
    
    granite = GraniteRAGClient(api_key=API_KEY, url=URL)

    print("🧠 Granite RAG Client Test")
    query_text = "Explain how adaptive prosthetic feedback reduces phantom pain."
    response = granite.query(query_text)
    print(f"Query: {query_text}")
    print(f"Response: {response}")
