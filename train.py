import os

# --- CONFIG: Redirect Cache to D: Drive ---
cache_dir = r"D:\Divyansh\huggingface_cache"
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
# ------------------------------------------

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    classification_report
)

# Using DistilBERT
model_name = "distilbert-base-uncased"
output_dir = "./model_output"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # 1. Load WELFake (0=Fake, 1=True)
    try:
        print("Loading Global Data...")
        ds = load_dataset("davanstrien/WELFake", split="train")
        df = pd.DataFrame(ds)[['text', 'label']]
        # Map: 1->0 (Real), 0->1 (Fake)
        df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)
    except: return

    # 2. Load Indian Data
    indian_path = os.path.join("data", "indian_news.csv")
    if os.path.exists(indian_path):
        print("Merging Indian Data...")
        try:
            df_in = pd.read_csv(indian_path)
            df_in.rename(columns=lambda x: x.strip().lower(), inplace=True)
            if df_in['label'].dtype == 'object':
                df_in['label'] = df_in['label'].apply(lambda x: 1 if 'FAKE' in str(x).upper() else 0)
            df = pd.concat([df, df_in[['text', 'label']]], ignore_index=True)
        except: pass

    # 3. Balance Data
    df = df.dropna()
    c0 = len(df[df['label'] == 0]) # Real
    c1 = len(df[df['label'] == 1]) # Fake
    min_c = min(c0, c1)
    
    print(f"Balancing to {min_c} samples each...")
    df_bal = pd.concat([
        df[df['label'] == 0].sample(n=min_c, random_state=42),
        df[df['label'] == 1].sample(n=min_c, random_state=42)
    ]).sample(frac=1).reset_index(drop=True)

    # 4. Train
    hf_ds = Dataset.from_pandas(df_bal).train_test_split(test_size=0.15, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized = hf_ds.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir, learning_rate=2e-5, 
        per_device_train_batch_size=16, per_device_eval_batch_size=16,
        num_train_epochs=2, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        fp16=True, save_total_limit=1, report_to="none", logging_steps=50
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"], tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    print("\nStarting Training...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model Saved.")

    # ---------------------------------------------------------
    # 5. GENERATE GRAPHS
    # ---------------------------------------------------------
    print("\n--- Generating Visualizations ---")
    
    # Get Predictions
    predictions = trainer.predict(tokenized["test"])
    logits = predictions.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    y_true = predictions.label_ids

    # A. Confusion Matrix [Image of Confusion Matrix]
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    print("Saved: confusion_matrix.png")
    plt.close()

    # B. ROC Curve [Image of ROC Curve]
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("Saved: roc_curve.png")
    plt.close()

    # C. Precision-Recall Curve [Image of Precision-Recall Curve]
    precision, recall, _ = precision_recall_curve(y_true, probs[:, 1])
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('pr_curve.png')
    print("Saved: pr_curve.png")
    plt.close()

    # D. Metrics Bar Chart
    metrics = {
        'Accuracy': accuracy_score(y_true, preds),
        'Precision': precision_recall_fscore_support(y_true, preds, average='binary')[0],
        'Recall': precision_recall_fscore_support(y_true, preds, average='binary')[1],
        'F1-Score': precision_recall_fscore_support(y_true, preds, average='binary')[2]
    }
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1.1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig('metrics_summary.png')
    print("Saved: metrics_summary.png")
    plt.close()

    # E. Training History
    history = trainer.state.log_history
    loss = [x['loss'] for x in history if 'loss' in x]
    if loss:
        plt.figure(figsize=(10, 5))
        plt.plot(loss, label='Training Loss')
        plt.title('Training Loss History')
        plt.xlabel('Steps (x50)')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_history.png')
        print("Saved: training_history.png")
        plt.close()

    print("\nAll Visualizations Generated Successfully!")

if __name__ == "__main__":
    main()