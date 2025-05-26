# .\env\Scripts\activate

"""
Fine Tuning

python finetuning/train.py --results_dir "results/msrvtt_valuation"

models/summary_generator.py
# Pretrained
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Fine-Tuned
model_path = "finetuning/models/flan-t5-msrvtt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

"""
import torch

if torch.cuda.is_available():
    print("✅ CUDA is available! GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ CUDA not available. Using CPU.")