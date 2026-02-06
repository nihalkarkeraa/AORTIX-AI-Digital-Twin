class Agent3Safety:
    def validate(self, sim_outputs):
        safe = {}

        for treatment, parts in sim_outputs.items():
            risky = any("High risk" in v for v in parts.values())
            if not risky:
                safe[treatment] = parts

        return safe
