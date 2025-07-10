#import sys
#import os
#sys.path.append('/content/pii_redactor/scripts')

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from preprocess import load_dataset, encode_labels, get_hf_dataset

model_name = "distilbert-base-cased"
MAX_LEN = 512
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["label_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx

        # Pad to MAX_LEN
        label_ids = label_ids[:MAX_LEN]
        label_ids += [-100] * (MAX_LEN - len(label_ids))
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    df = load_dataset("/content/pii_redactor/data/pii_dataset.csv")
    df, label_encoder = encode_labels(df)
    dataset = get_hf_dataset(df)

    label_list = list(label_encoder.classes_)  # 👈 Fix: define label_list
    dataset = dataset.map(tokenize, batched=True, remove_columns=["tokens", "label_ids"])

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_list)
    )

    args = TrainingArguments(
        output_dir="/content/models/pii_ner_model",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir="/content/logs",
        logging_steps=50,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save model & tokenizer
    model.save_pretrained("/content/models/pii_ner_model")
    tokenizer.save_pretrained("/content/models/pii_ner_model")

    # Save label list
    with open("/content/models/pii_ner_model/labels.txt", "w") as f:
        for label in label_list:
            f.write(f"{label}\n")

if __name__ == "__main__":
    main()
