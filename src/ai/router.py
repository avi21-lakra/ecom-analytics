from .metrics import compute_metrics

def route_intent(intent: str):
    metrics = compute_metrics()

    if intent in metrics:
        return metrics[intent]

    return metrics
