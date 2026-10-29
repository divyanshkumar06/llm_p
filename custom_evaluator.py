import numpy as np

class CustomEvaluator:
    """
    Evaluator for classification models built strictly from scratch without sklearn.
    
    Computes:
    - Confusion Matrix (C x C)
    - Per-class Precision, Recall, and F1-Score
    - Overall Accuracy
    - Macro-F1 Evaluation Metric
    """
    @staticmethod
    def evaluate(y_true, y_pred, labels=None):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        else:
            labels = np.array(labels)
            
        n_classes = len(labels)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        # Build Confusion Matrix (C x C)
        # Entry [i, j] = count of samples with true class i predicted as class j
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                cm[label_to_idx[t], label_to_idx[p]] += 1
                
        # Per-class metrics
        precision_list = []
        recall_list = []
        f1_list = []
        support_list = []
        
        per_class_results = {}
        
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            support = np.sum(cm[i, :])
            
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
            
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            support_list.append(support)
            
            per_class_results[label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(support)
            }
            
        # Macro Metrics
        macro_precision = float(np.mean(precision_list))
        macro_recall = float(np.mean(recall_list))
        macro_f1 = float(np.mean(f1_list))
        
        # Accuracy & Micro Metrics
        # In multi-class classification, Micro Precision = Micro Recall = Micro F1 = Overall Accuracy
        total_samples = len(y_true)
        correct_samples = int(np.sum(y_true == y_pred))
        accuracy = float(correct_samples / total_samples) if total_samples > 0 else 0.0
        micro_precision = accuracy
        micro_recall = accuracy
        micro_f1 = accuracy
        
        return {
            "accuracy": accuracy,
            "micro_f1": micro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class": per_class_results,
            "confusion_matrix": cm,
            "labels": labels.tolist()
        }

    @staticmethod
    def print_classification_report(metrics_dict):
        """
        Prints a clean formatted classification table showing both Micro-F1 and Macro-F1.
        """
        print("\n" + "="*70)
        print("                   CLASSIFICATION EVALUATION REPORT                   ")
        print("="*70)
        print(f"{'Class Label':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print("-" * 70)
        
        for label, m in metrics_dict["per_class"].items():
            print(f"{label:<15} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['support']:<8d}")
            
        total_support = sum(m['support'] for m in metrics_dict['per_class'].values())
        print("-" * 70)
        print(f"{'Micro Average':<15} | {metrics_dict['micro_precision']:<10.4f} | {metrics_dict['micro_recall']:<10.4f} | {metrics_dict['micro_f1']:<10.4f} | {total_support:<8d}")
        print(f"{'Macro Average':<15} | {metrics_dict['macro_precision']:<10.4f} | {metrics_dict['macro_recall']:<10.4f} | {metrics_dict['macro_f1']:<10.4f} | {total_support:<8d}")
        print(f"{'Accuracy':<15} | {'':<10} | {'':<10} | {metrics_dict['accuracy']:<10.4f} | {total_support:<8d}")
        print("=" * 70)
        print(f"\n>>> MICRO-F1 EVALUATION METRIC SCORE: {metrics_dict['micro_f1'] * 100:.2f}% <<<")
        print(f">>> MACRO-F1 EVALUATION METRIC SCORE: {metrics_dict['macro_f1'] * 100:.2f}% <<<\n")

if __name__ == "__main__":
    # Sanity check
    y_t = ["hin_Deva", "hin_Deva", "tam_Taml", "ben_Beng"]
    y_p = ["hin_Deva", "tam_Taml", "tam_Taml", "ben_Beng"]
    res = CustomEvaluator.evaluate(y_t, y_p)
    CustomEvaluator.print_classification_report(res)
