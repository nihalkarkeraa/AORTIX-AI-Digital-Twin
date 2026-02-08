import requests

class Agent5Summary:

    def generate(self, agent1, agent2, agent3, agent4):

        prompt = f"""
You are AORTIX, a cardiovascular AI assistant.

System outputs:

Agent1:
{agent1}

Agent2:
{agent2}

Agent3:
{agent3}

Agent4:
{agent4}

Generate a simple clinical summary including:
- Patient condition
- Final treatment
- Risk level
- Confidence
- Monitoring advice
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]
