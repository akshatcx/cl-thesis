from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset
from transformers import AutoTokenizer
import spacy
from string import punctuation
import random
from itertools import chain
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import torch
import numpy as np
from transformers import Pipeline

SENTS_TO_COMBINE = 3
DATASET_SIZE = 0.01
SHOW_N_SCORES = 5
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "../models/proto1"
EPOCHS = 4
BATCH_SIZE = 64

GENERAL_DOMAIN = 'gen'

label_list = []
id2label = {}
label2id = {}

nlp = spacy.load('en_core_web_md')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")

def fetch_20news():
    newsgroups = fetch_20newsgroups(subset='all')

    global label_list
    global id2label
    global label2id
    label_list = newsgroups.target_names
    label_list.append(GENERAL_DOMAIN)
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {v: k for k, v in id2label.items()}
    
    dataset = Dataset.from_dict({'text': newsgroups.data, 'target': newsgroups.target})
    return dataset.shuffle().shard(num_shards=1/DATASET_SIZE, index=0)
 
def generateLabels(row):
    text = row['text']
    text = text.replace('\n', ' ')
    
    doc = nlp(text)

    sents = []
    for sent in doc.sents:
        lsent = []
        for w in sent:
            if w.text.isspace(): continue
            elif w.text in punctuation or w.is_stop: lsent.append([w.text, label2id[GENERAL_DOMAIN]])
            else: lsent.append([w.text, row['target']])
        sents.append(list(map(list, zip(*lsent))))
    return sents

def create_dataset(raw_dataset: Dataset):
    sents = []
    for row in raw_dataset: sents.extend(generateLabels(row))

    # combine sents 
    tokens = []
    tags = []
    random.shuffle(sents)
    csents = [sents[n:n+SENTS_TO_COMBINE] for n in range(0, len(sents), SENTS_TO_COMBINE)]
    for i in range(len(csents)):
        k = [list(chain.from_iterable(x)) for x in zip(*csents[i])]
        if(len(k) == 2):
            tokens.append(k[0])
            tags.append(k[1])

    return Dataset.from_dict({'tokens': tokens, 'tags': tags}).train_test_split(test_size=0.1)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# main
raw_dataset = fetch_20news()

d = create_dataset(raw_dataset)
dd = d.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
   MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
    use_mps_device=torch.backends.mps.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dd["train"],
    eval_dataset=dd["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()

print(f"training complete, model saved to {OUTPUT_DIR}")


# inference pipeline
# class DTEPipeline(Pipeline):
#     def _sanitize_parameters(self, **kwargs):
#         preprocess_kwargs = {}
#         return preprocess_kwargs, {}, {}

#     def preprocess(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt")
#         return inputs

#     def _forward(self, model_inputs):
#         outputs = self.model(**model_inputs)
#         return {
#             "logits": outputs.logits,
#             **model_inputs,
#         }

#     def postprocess(self, model_outputs):
#         input_ids = model_outputs["input_ids"][0]
#         l = model_outputs["logits"][0].softmax(-1).numpy()
        
#         tokens = []
#         for i in range(len(input_ids)):
#             tokens.append({
#                 "word": self.tokenizer.convert_ids_to_tokens(int(input_ids[i])),
#                 "tag": id2label[np.argmax(l[i])],
#                 "topk_scores": list(np.round(np.sort(np.partition(l[i], -SHOW_N_SCORES)[-SHOW_N_SCORES:])[::-1], 4))
#             })
#         return tokens
    
# classifier = DTEPipeline(model = model, tokenizer = tokenizer)
# classifier("this is a sentence")
