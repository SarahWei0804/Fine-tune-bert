This repository provides Python scripts to fine-tune Google’s BERT model for classifying MITRE ATT&CK Enterprise tactics(v16) from a given sentence. By training on labeled attack descriptions, the model helps identify how adversaries execute attacks, supporting threat intelligence and incident response.

## Contents
- fine-tune_bert.py – Script for fine-tuning BERT on MITRE ATT&CK tactic classification.
- requirements.txt – List of dependencies needed to run the script.

## Usage
Install dependencies

```bash=
pip install -r requirements.txt
```

Run the fine-tuning script

```bash
python fine-tune_bert.py
```
This model enhances automated threat detection by accurately mapping text to MITRE ATT&CK tactics(v16). 
