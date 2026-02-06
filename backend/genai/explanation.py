def generate_explanation(agent1, decision):
    summary = "AI analysis of the digital heart twin indicates:\n"

    for part, status in agent1.items():
        summary += f"- {part.replace('_',' ').title()}: {status}\n"

    summary += f"\nRecommended treatment: {decision['decision']} "
    summary += f"(confidence {decision['confidence']})"

    return summary
