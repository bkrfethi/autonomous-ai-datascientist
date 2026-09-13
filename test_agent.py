from agent import run_autonomous_agent

report = run_autonomous_agent(
    csv_path="uploads/churn_data.csv",
    target_col="churn",
    query="Quels sont les facteurs majeurs du churn client et comment réduire les départs ?"
)

print(report)