import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def main():
    print("Loading datasets...")
    fake_df = pd.read_csv("Dataset/Fake.csv")
    true_df = pd.read_csv("Dataset/True.csv")
    indian_df = pd.read_csv("Dataset/news_dataset.csv")

    fake_df["label"] = 1
    true_df["label"] = 0
    fake_df["final_text"] = fake_df["title"].astype(str) + " " + fake_df["text"].astype(str)
    true_df["final_text"] = true_df["title"].astype(str) + " " + true_df["text"].astype(str)
    
    indian_df["final_text"] = indian_df["text"].astype(str)
    indian_df["label"] = indian_df["label"].replace({"REAL": 0, "FAKE": 1})

    fake_final = fake_df[["final_text", "label"]]
    true_final = true_df[["final_text", "label"]]
    indian_final = indian_df[["final_text", "label"]]

    combined_df = pd.concat([fake_final, true_final, indian_final], ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.rename(columns={"final_text": "text"})
    
    print("Cleaning text...")
    combined_df["clean_text"] = combined_df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        combined_df["clean_text"], combined_df["label"],
        test_size=0.2, random_state=42, stratify=combined_df["label"]
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_dataset = Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"])
    test_dataset = Dataset.from_pandas(test_df).remove_columns(["__index_level_0__"])

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    train_tokenized = train_tokenized.rename_column("label", "labels")
    test_tokenized = test_tokenized.rename_column("label", "labels")

    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results_v2",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=200,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics
    )

    print("Training model...")
    trainer.train()

    print("Saving model to distilbert_v2_indian_model...")
    model.save_pretrained("distilbert_v2_indian_model")
    tokenizer.save_pretrained("distilbert_v2_indian_model")
    print("Done!")

if __name__ == "__main__":
    main()
