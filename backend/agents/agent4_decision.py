class Agent4Decision:
    def choose(self, safe_outputs):
        if not safe_outputs:
            return {"decision": "No safe treatment", "confidence": 0.0}

        best = list(safe_outputs.keys())[0]
        return {
            "decision": best,
            "confidence": 0.82
        }
