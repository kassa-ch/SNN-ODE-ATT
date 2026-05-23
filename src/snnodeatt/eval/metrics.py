import numpy as np

def binary_metrics(y_true, scores, threshold, direction="high"):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores > threshold).astype(int) if direction == "high" else (scores < threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    f2 = 5 * precision * recall / max(1e-12, 4 * precision + recall)
    return dict(accuracy=(tp + tn) / max(1, len(y_true)), precision=precision,
                recall=recall, f1=f1, f2=f2, tp=tp, fp=fp, tn=tn, fn=fn)
