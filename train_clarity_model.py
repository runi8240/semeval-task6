import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import evaluate


# 1. Load the dataset
print("Loading dataset...")
ds = load_dataset("ailsntua/QEvasion")

# Filter out examples with missing questions or answers
ds = ds.filter(lambda x: x["interview_question"] is not None and x["interview_answer"] is not None)

# View label names and map labels to IDs
clarity_labels = sorted(set(ds["train"]["clarity_label"]))
evasion_labels = sorted(set(ds["train"]["evasion_label"]))

clarity2id = {c: i for i, c in enumerate(clarity_labels)}
id2clarity = {i: c for c, i in clarity2id.items()}
evasion2id = {c: i for i, c in enumerate(evasion_labels)}
id2evasion = {i: c for c, i in evasion2id.items()}

print("Clarity labels:", clarity2id)
print("Evasion labels:", evasion2id)

# 2. Tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def preprocess(batch):
    """Combine question + answer and tokenize."""
    questions = [q or "" for q in batch["interview_question"]]
    answers = [a or "" for a in batch["interview_answer"]]

    texts = [q + " [SEP] " + a for q, a in zip(questions, answers)]
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=512)

    encodings["labels_clarity"] = [clarity2id[label] if label in clarity2id else 0 for label in batch["clarity_label"]]
    encodings["labels_evasion"] = [evasion2id[label] if label in evasion2id else 0 for label in batch["evasion_label"]]

    return encodings


print("Tokenizing dataset...")
encoded = ds.map(preprocess, batched=True)

# Remove all non-model columns
encoded = encoded.remove_columns(ds["train"].column_names)

# Set tensor format
encoded.set_format("torch")


# 3. Multi-task model
class MultiTaskDeberta(nn.Module):
    def __init__(self, base_model, num_clarity, num_evasion):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model, use_safetensors=True)
        hidden = self.encoder.config.hidden_size

        # Two classifier heads
        self.clarity_head = nn.Linear(hidden, num_clarity)
        self.evasion_head = nn.Linear(hidden, num_evasion)
    
    def forward(self, input_ids, attention_mask, labels_clarity=None, labels_evasion=None):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = enc_out.last_hidden_state[:, 0, :]    # CLS token embedding

        clarity_logits = self.clarity_head(cls)
        evasion_logits = self.evasion_head(cls)

        loss = None
        if labels_clarity is not None and labels_evasion is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = (loss_fct(clarity_logits, labels_clarity) + loss_fct(evasion_logits, labels_evasion))
        
        return {"loss": loss, "logits_clarity": clarity_logits, "logits_evasion": evasion_logits}


print("Initializing model...")
model = MultiTaskDeberta(base_model=model_name, num_clarity=len(clarity2id), num_evasion=len(evasion2id))

# 4. Training
training_args = TrainingArguments(
    output_dir="./clarity_model",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none"
)


def compute_metrics(pred):
    clarity_logits = pred.predictions[0]
    evasion_logits = pred.predictions[1]

    labels = pred.label_ids
    clarity_labels = labels[:, 0]
    evasion_labels = labels[:, 1]

    clarity_preds = clarity_logits.argmax(axis=-1)
    evasion_preds = evasion_logits.argmax(axis=-1)

    acc = evaluate.load("accuracy")

    return {
        "clarity_accuracy": acc.compute(predictions=clarity_preds, references=clarity_labels),
        "evasion_accuracy": acc.compute(predictions=evasion_preds, references=evasion_labels
        )
    }


trainer = Trainer(model=model, args=training_args, train_dataset=encoded["train"], eval_dataset=encoded.get("validation", encoded["test"]), compute_metrics=compute_metrics)

print("\nStarting training...\n")
trainer.train()

# 5. Save the model
print("Saving model...")
trainer.save_model("./clarity_model")


# 6. Inference function
def predict(question, answer):
    question = question or ""
    answer = answer or ""
    text = question + " [SEP] " + answer

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model(**inputs)
    
    c_id = output["logits_clarity"].argmax(dim=-1).item()
    e_id = output["logits_evasion"].argmax(dim=-1).item()

    return id2clarity[c_id], id2evasion[e_id]


# Example
print("\nExample prediction:")
q = "Why did you veto the bill?"
a = "Because the timing was not appropriate."
print(predict(q, a))
