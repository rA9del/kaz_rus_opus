import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import argparse
import os
import pandas as pd
import polars as pl
import torch
import evaluate
from datasets import Dataset
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer,DataCollatorForSeq2Seq



from tqdm import tqdm

if torch.cuda.is_available():
    print(f"🔹 Using GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})")

# ----------------- Load and Filter -----------------
def load_data(csv_path, score_threshold=0.0):
    df = pl.read_csv(csv_path, separator="\t")
    df = df.filter(pl.col("score") >= score_threshold)
    # df = pd.read_csv(csv_path, sep="\t")
    # df = df[df["score"] >= score_threshold]
    return df.to_pandas()

# ----------------- Analyze Lengths -----------------
def analyze_lengths(df, tokenizer, max_length=128):
    lengths = []
    for text in tqdm(df["kazakh"].to_list() + df["russian"].to_list(), total=len(df)*2):
        tokens = tokenizer(text, truncation=True, max_length=max_length).input_ids
        lengths.append(len(tokens))
    print(f"🔹 95th percentile of token lengths: {np.percentile(lengths, 95)}")
    return lengths

# ----------------- Tokenize -----------------
def tokenize_data(df, tokenizer, max_length=128, cache_dir=None):
    def preprocess(examples):
        inputs = [f"translate Kazakh to Russian: {text}" for text in examples["kazakh"]]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["russian"], max_length=max_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_pandas(df[["kazakh", "russian"]])
    print(f"🔹 Tokenizing dataset ({len(dataset)} samples)...")
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["kazakh", "russian"])
    return tokenized_dataset


# ----------------- Training -----------------
def train_model(train_ds, val_ds, tokenizer, model_name, output_dir, epochs, batch_size):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.gradient_checkpointing_enable()

    training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            dataloader_num_workers=4,
            eval_strategy="no",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=5e-5,
            num_train_epochs=epochs,
            save_total_limit=2,
            logging_dir=f"{output_dir}/logs",
            predict_with_generate=True,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_accumulation_steps=16,
            logging_strategy="epoch",
            report_to="none",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model)
    )

    train_output = trainer.train()

    metrics_df = pd.DataFrame(trainer.state.log_history)
    metrics_df.to_csv(os.path.join(output_dir, "training_metrics.csv"), index=False)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# ----------------- BLEU Evaluation -----------------
def evaluate_bleu(model, tokenizer, dataset, max_len=128):
    bleu = evaluate.load("sacrebleu")
    for example in dataset.select(range(100)):
        inputs = tokenizer(example["kazakh"], return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=max_len)
        pred = tokenizer.decode(generated[0], skip_special_tokens=True)
        bleu.add(prediction=pred, reference=[example["russian"]])
    score = bleu.compute()
    print("🔹 BLEU Score:", score)
    
    



# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to train TSV")
    parser.add_argument("--test", type=str, required=True, help="Path to test TSV")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="./mt_model")
    parser.add_argument("--model", type=str, default="facebook/m2m100_418M")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)  # Increased for efficiency
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for dataset cache")
    args = parser.parse_args()

    print("🔹 Loading and filtering data...")
    train_set = load_data(args.train, args.score_threshold)
    test_set = load_data(args.test, 0.65)

    print(f"🔹 Loaded {len(train_set)} filtered train pairs (score ≥ {args.score_threshold})")
    print(f"🔹 Loaded {len(test_set)} filtered test pairs (score ≥ 0.65)")

    print(f"🔹 Loading model {args.model}...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenizer.model_max_length = args.max_len

    tokenizer.src_lang = "kk"

    # print("🔹 Analyzing token lengths...")
    # analyze_lengths(train_set, tokenizer, args.max_len)

    print("🔹 Tokenizing data...")
    train_dataset = tokenize_data(train_set, tokenizer, max_length=args.max_len, cache_dir=args.cache_dir)
    test_dataset = tokenize_data(test_set, tokenizer, max_length=args.max_len, cache_dir=args.cache_dir)

    print("🔹 Starting training...")
    train_model(train_dataset, test_dataset, tokenizer, args.model, args.output_dir, args.epochs, args.batch_size)

    print("🔹 Evaluating BLEU...")
    model = T5ForConditionalGeneration(config=T5ForConditionalGeneration.config_class.from_pretrained("t5-small"))
    tokenizer.src_lang = "kk"
    evaluate_bleu(model, tokenizer, test_dataset, max_len=args.max_len)