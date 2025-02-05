from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, TrainingArguments, BertForSequenceClassification, Trainer
import os
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset=load_dataset("sarahwei/cyber_MITRE_tactic_CTI_dataset_v16")

## transform the string to list
tactic_trans = []
for i in dataset['train']['label']:
    tactic_trans.append(ast.literal_eval(i))

dataset['train'] = dataset['train'].add_column("tactic", tactic_trans)

## labeling
mlb = MultiLabelBinarizer()
label_num = mlb.fit_transform(dataset['train']['tactic'])
label2id = dict(zip(mlb.classes_, [i for i in range(len(mlb.classes_))]))
id2label = dict(zip([i for i in range(len(mlb.classes_))], mlb.classes_))
dataset_embed = datasets.Dataset.from_dict({"tactic_label": label_num})
dataset_concat = datasets.concatenate_datasets([dataset['train'], dataset_embed], axis=1)


tokenizer = AutoTokenizer.from_pretrained("bencyc1129/mitre-bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset_concat.map(tokenize_function, batched=True)


dataset = tokenized_datasets.train_test_split(test_size=0.2)
## remove columns and rename
dataset = dataset.remove_columns(["text", "label", "tactic"])
dataset = dataset.rename_column("tactic_label", "labels")
train_dataset = dataset["train"]
test_dataset = dataset["test"]
num_labels=len(label2id)
model = BertForSequenceClassification.from_pretrained("bencyc1129/mitre-bert-base-cased", label2id = label2id, id2label=id2label, num_labels=num_labels)

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss
    

args = TrainingArguments(
    output_dir="bert_cased_trainer",
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_steps=500,
    logging_steps=500,
    optim="adamw_8bit",
    seed=0
)
def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True): 
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid: 
      y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'accuracy': accuracy_thresh(predictions, labels)}

multi_trainer = MultilabelTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer)

multi_trainer.evaluate()
## start fine tune
multi_trainer.train()

## save the fine-tuned model
multi_trainer.save_model("./MITRE-tactic-v16-bert-cased")