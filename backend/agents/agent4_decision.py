class Agent4Decision:

    def decide(self, agent3_output):

        status = agent3_output["status"]
        confidence = agent3_output["confidence"]
        risk_level = agent3_output["risk_level"]
        best = agent3_output["best_treatment"]

        # Default response
        decision = "PENDING"
        strategy = "Manual clinical review required"
        monitoring = "Immediate observation"

        # SAFE CASE
        if status == "SAFE" and confidence >= 0.3:
            decision = "APPROVED"
            strategy = f"Proceed with {best}"
            monitoring = "Standard monitoring"

        # WARNING CASE
        elif status == "WARNING":
            decision = "CAUTION"
            strategy = f"Proceed with {best} under supervision"
            monitoring = "Monitor vitals every 4 hours"

        # DANGEROUS CASE
        elif status == "DANGEROUS":
            decision = "REJECTED"
            strategy = "Treatment rejected – manual intervention required"
            monitoring = "Continuous monitoring"

        return {
            "final_treatment": best if decision != "REJECTED" else "NONE",
            "decision": decision,
            "strategy": strategy,
            "monitoring": monitoring,
            "confidence": confidence,
            "risk_level": risk_level
        }
