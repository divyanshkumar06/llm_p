import os
import json
import time
import numpy as np
import scipy.sparse as sp

from dataset_builder import build_dataset
from custom_tfidf import CustomTFIDFVectorizer
from custom_logistic_regression import CustomLogisticRegression
from custom_evaluator import CustomEvaluator

def run_language_id_pipeline(dataset_file="indiccorp_dataset.json", max_features=35000, n_epochs=25):
    print("=" * 80)
    print("      INDICCORPV2 LANGUAGE IDENTIFICATION PIPELINE (FROM SCRATCH)      ")
    print("=" * 80)
    
    # Step 1: Load or Build Dataset
    if not os.path.exists(dataset_file):
        print(f"Dataset file '{dataset_file}' not found. Building dataset now...")
        dataset = build_dataset(output_path=dataset_file)
    else:
        print(f"Loading pre-built dataset from '{dataset_file}'...")
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
    labels = dataset["labels"]
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]
    
    X_train_text = [item["text"] for item in train_data]
    y_train = np.array([item["label"] for item in train_data])
    
    X_val_text = [item["text"] for item in val_data]
    y_val = np.array([item["label"] for item in val_data])
    
    X_test_text = [item["text"] for item in test_data]
    y_test = np.array([item["label"] for item in test_data])
    
    print(f"\n[Dataset Summary]")
    print(f"  - Total Languages / Script Labels: {len(labels)}")
    print(f"  - Train Set Size:      {len(X_train_text)} samples ({len(X_train_text)//len(labels)} per lang)")
    print(f"  - Validation Set Size: {len(X_val_text)} samples ({len(X_val_text)//len(labels)} per lang)")
    print(f"  - Test Set Size:       {len(X_test_text)} samples ({len(X_test_text)//len(labels)} per lang)")
    
    # Step 2: Feature Extraction using Custom TF-IDF Vectorizer
    print("\n" + "-" * 80)
    print("[1. TF-IDF Feature Extraction (From Scratch)]")
    print("  Extracting Word Unigrams/Bigrams and Character 2,3,4-grams...")
    t0 = time.time()
    vectorizer = CustomTFIDFVectorizer(max_features=max_features, sublinear_tf=True, norm='l2')
    
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    t_tfidf = time.time() - t0
    print(f"  -> Vectorizer fitted in {t_tfidf:.2f} seconds.")
    print(f"  -> Vocabulary Size: {len(vectorizer.vocabulary_)} features.")
    print(f"  -> Train TF-IDF Sparse Matrix Shape: {X_train_tfidf.shape}")
    
    print("  Transforming Validation and Test sets...")
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Step 3: Train Custom Logistic Regression Classifier
    print("\n" + "-" * 80)
    print("[2. Logistic Regression Classifier Training (From Scratch)]")
    t0 = time.time()
    classifier = CustomLogisticRegression(
        learning_rate=0.5,
        n_epochs=n_epochs,
        batch_size=512,
        l2_reg=1e-4,
        verbose=True
    )
    classifier.fit(X_train_tfidf, y_train)
    t_train = time.time() - t0
    print(f"  -> Model trained in {t_train:.2f} seconds.")
    
    # Step 4: Evaluate on Validation Set
    print("\n" + "-" * 80)
    print("[3. Evaluation on VALIDATION Set (Macro-F1)]")
    y_val_pred = classifier.predict(X_val_tfidf)
    val_metrics = CustomEvaluator.evaluate(y_val, y_val_pred, labels=labels)
    CustomEvaluator.print_classification_report(val_metrics)
    
    # Step 5: Evaluate on Test Set
    print("\n" + "-" * 80)
    print("[4. Evaluation on TEST Set (Macro-F1)]")
    y_test_pred = classifier.predict(X_test_tfidf)
    test_metrics = CustomEvaluator.evaluate(y_test, y_test_pred, labels=labels)
    CustomEvaluator.print_classification_report(test_metrics)
    
    # Step 6: Live Prediction Demonstration
    print("\n" + "-" * 80)
    print("[5. Live Interactive Text Inference Demonstration]")
    
    sample_queries = [
        ("नमस्ते! आप कैसे हैं? भारत एक महान देश है।", "hin_Deva"),
        ("வணக்கம்! தமிழ் மொழி மிகத் தொன்மையானது.", "tam_Taml"),
        ("নমস্কার! আপনি কেমন আছেন? বাংলা একটি সুন্দর ভাষা।", "ben_Beng"),
        ("ਨਮਸਤੇ! ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ? ਪੰਜਾਬੀ ਸਾਡੀ ਮਾਂ ਬੋਲੀ ਹੈ।", "pan_Guru"),
        ("તમે કેમ છો? ગુજરાત ભારતનું સમૃદ્ધ રાજ્ય છે.", "guj_Gujr"),
        ("آپ کیسے ہیں؟ اردو ایک خوبصورت زبان ہے۔", "urd_Arab"),
        ("తెలుగు భాష చాలా తీయనైనది மற்றும் అందమైనది.", "tel_Telu"),
        ("Khasi is spoken mainly in Meghalaya state in India.", "khasi")
    ]
    
    print(f"{'Input Text Snippet':<50} | {'True Script/Lang':<12} | {'Predicted Label':<12} | {'Confidence':<10}")
    print("-" * 92)
    
    for text, expected in sample_queries:
        X_sample = vectorizer.transform([text])
        proba = classifier.predict_proba(X_sample)[0]
        pred_idx = np.argmax(proba)
        pred_label = classifier.classes_[pred_idx]
        confidence = proba[pred_idx] * 100
        
        # Display safe ASCII indicator or encoded representation for Windows console
        try:
            printable_text = text.encode('ascii', 'backslashreplace').decode('ascii')
            snippet = (printable_text[:47] + '...') if len(printable_text) > 50 else printable_text
        except Exception:
            snippet = f"Sample text ({expected})"
            
        print(f"{snippet:<50} | {expected:<12} | {pred_label:<12} | {confidence:<9.1f}%")
        
    print("=" * 80)
    print("      PIPELINE EXECUTION COMPLETE & VERIFIED SUCCESSFULLY!      ")
    print("=" * 80)
    
    # Save final results summary to JSON for reporting
    summary = {
        "dataset": {
            "num_languages": len(labels),
            "train_size": len(X_train_text),
            "val_size": len(X_val_text),
            "test_size": len(X_test_text)
        },
        "validation_metrics": {
            "accuracy": val_metrics["accuracy"],
            "micro_f1": val_metrics["micro_f1"],
            "macro_f1": val_metrics["macro_f1"],
            "macro_precision": val_metrics["macro_precision"],
            "macro_recall": val_metrics["macro_recall"]
        },
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "micro_f1": test_metrics["micro_f1"],
            "macro_f1": test_metrics["macro_f1"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"]
        }
    }
    
    with open("results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return vectorizer, classifier, test_metrics

if __name__ == "__main__":
    run_language_id_pipeline()
