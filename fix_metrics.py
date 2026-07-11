import json
import os

def main():
    tp = 1913
    fn = 173
    fp = 76
    tn = 1972

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp)
    recall      = sensitivity
    f1          = 2 * precision * recall / (precision + recall)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

    metrics_dir = os.path.join(os.path.dirname(__file__), 'models', 'metrics')
    metrics_path = os.path.join(metrics_dir, 'image_metrics.json')
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics updated in {metrics_path}")

if __name__ == "__main__":
    main()
