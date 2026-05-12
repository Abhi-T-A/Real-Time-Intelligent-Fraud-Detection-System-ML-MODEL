import pandas as pd

def build_graph(df):
    graph = {}

    for _, row in df.iterrows():
        user = f"user_{_}"
        payee = f"payee_{row['tx_to_payee']}"

        if payee not in graph:
            graph[payee] = {
                "transactions": 0,
                "fraud_count": 0
            }

        graph[payee]["transactions"] += 1

        if row["is_fraud"] == 1:
            graph[payee]["fraud_count"] += 1

    return graph


def compute_graph_risk(graph, payee_id):
    payee = f"payee_{payee_id}"

    if payee not in graph:
        return 0.1  # low risk

    fraud_ratio = graph[payee]["fraud_count"] / (graph[payee]["transactions"] + 2)

    return min(fraud_ratio, 1.0)