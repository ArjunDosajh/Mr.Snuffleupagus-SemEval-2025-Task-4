# Mr. Snuffleupagus at [Semeval 2025 Task 4: Unlearning Sensitive Content from Large Language Models](https://llmunlearningsemeval2025.github.io/)

This repository contains the submission for **Semeval 2025 Task 4: Unlearning Sensitive Content from Large Language Models** by the team *Mr. Snuffleupagus*. The objective of this task is to remove specific knowledge from a pretrained language model while preserving its general capabilities.

---

## 🚀 Running the Unlearning Script

The script `unlearn_submission.py` is used to perform the unlearning process. Below is the command to execute it:

```bash
python unlearn_submission.py \
    --model_path allenai/OLMo-7B \
    --output_dir /path/to/output \
    --forget_dataset_path data/forget.parquet \
    --retain_dataset_path data/retain.parquet
```

### Arguments:
- `--model_path`: Path to the pretrained model (e.g., `allenai/OLMo-7B`).
- `--output_dir`: Directory to store the modified model.
- `--forget_dataset_path`: Path to the dataset containing information to be forgotten.
- `--retain_dataset_path`: Path to the dataset containing information to be retained.

---

## 📂 Directory Structure

```
├── data/
│   ├── forget.parquet    # Dataset containing sensitive content to be removed
│   ├── retain.parquet    # Dataset containing knowledge to be preserved
│
├── evaluate/
│   ├── run_mmlu.py               # MMLU Score
│   ├── evaluate_generations.py   # Task Aggregate Score
│
├── unlearn_submission.py  # Script to perform unlearning
├── README.md              # Documentation
```

---

## 🛠 Model Details
- **Base Model**: [OLMo-7B](https://huggingface.co/allenai/OLMo-7B)
- **Unlearning Method**: Adaptive Representation Misdirection Unlearning (RMU)

---

## 📋 Evaluation
To evaluate the model post-unlearning, use the following commands:

### Calculate MMLU
```bash
python run_eval.py \
    --data_dir data/ \
    --save_dir results/olmo7b/ \
    --model_name_or_path "allenai/OLMo-7B-0724-hf" \
    --tokenizer_name_or_path "allenai/OLMo-7B-0724-hf" \
    --eval_batch_size 8
```

### Calculate Task Aggregate Score
MIA data can be downloaded form the [SemEval 2025 Task 4 website](https://llmunlearningsemeval2025.github.io/)
```bash
python evaluate_generations.py \
    --data_path jsonl_data/ \
    --mia_data_path mia_data/ \
    --output_dir eval_output/ \
    --checkpoint_path /path/to/unlearned_model
```

---

## 📝 Citation

## 🔗 References
- [SemEval 2025 Task 4](https://llmunlearningsemeval2025.github.io/)
- [OLMo-7B Model](https://huggingface.co/allenai/OLMo-7B)

## Authors
- [Arjun Dosajh](https://github.com/ArjunDosajh)
- [Mihika Sanghi](https://github.com/mihikasanghi)
