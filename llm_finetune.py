from datasets import load_dataset
import pandas as pd
from iso639 import Lang
import torch
import transformers
import string 
import re
import json
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
#from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
# Parse command line arguments
import argparse
from huggingface_hub import login

login(
  token="", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)
parser = argparse.ArgumentParser(description="Train a model on a dataset")
parser.add_argument("--base_model", type=str,  help="sent level", default="CohereForAI/aya-expanse-8b")
parser.add_argument("--save_dir", type=str,  help="save directory for the model", default="c4ai")
parser.add_argument("--data_dir", type=str,  help="data directory", default="./")
parser.add_argument("--source_langs", type=str,  help="source languages", default="apc,aeb,arb,arq,arz,ary")
parser.add_argument("--lr", type=float,  help="lr", default=2e-4)
parser.add_argument("--r", type=int,  help="r", default=32)
parser.add_argument("--one_by_one", type=bool,  help="r", default=False)
parser.add_argument("--newest_data", type=bool,  help="r", default=False)

parser.add_argument("--restore", type=bool,  help="r", default=False)

args=parser.parse_args()
data_dir = args.data_dir
source_langs = args.source_langs.split(",")
GEN_LEN = "gen_len"
BLEU = "bleu"
df_train=pd.read_csv(f"{data_dir}/train.csv", dtype=str, skiprows=1, names=["Document ID","Copy ID","Dataset Type","Original Document","Masked Document","Masked Words"])
df_dev=pd.read_csv(f"{data_dir}/dev.csv" , skiprows=1, dtype=str, names=["Document ID","Copy ID","Dataset Type","Original Document","Masked Document","Masked Words"])



def create_conversation(sample, newest=False):
    if args.newest_data:
        message ={ "messages":[{"role": "user",
        "content":f"Fill in the missing {sample['Language'].capitalize()} words, masked by the [MASK] token. Output \"WORDS:\" and a comma-separated list of the missing words in original {sample['Language'].capitalize()}: {sample['Masked Document'].replace(' [NEWLINE] ','\n')}"},
        {"role": "assistant",
                   "content": f"WORDS: {sample['Masked Words']}"
                }]}

    else:
        message ={ "messages":[{"role": "user",
        "content":f"Fill in the missing Akkadian words, masked by the [MASK] token. Output \"WORDS:\" and a comma-separated list of the missing words in original Akkadian: {sample['Masked Document']}"},
        {"role": "assistant",
                   "content": f"WORDS: {sample['Masked Words']}"
                }]}

    return message
def create_conversation_single(document, word):
    if args.newest_data:
        message ={ "messages":[{"role": "user",
        "content":f"Fill in the missing {sample['Language'].capitalize()} word masked by the [MASK] token: {document.replace(' [NEWLINE] ','\n')}"},
        {"role": "assistant",
                   "content": f"{word}"
                }]}


    else:
        message ={ "messages":[{"role": "user",
        "content":f"Fill in the missing Akkadian word masked by the [MASK] token: {document}"},
        {"role": "assistant",
                   "content": f"{word}"
                }]}
    
    return message
def create_conversation_restore(sample):
    if args.newest_data:
        message ={ "messages":[{"role": "user",
        "content":f"Complete the missing {sample['Language'].capitalize()} words masked by the [MASK] tokens and print out the restored document: {sample['Masked Document'].replace(' [NEWLINE] ','\n')}"},
        {"role": "assistant",
                   "content": f"{sample['Original Document']}"
                }]}

    else:
        message ={ "messages":[{"role": "user",
        "content":f"Complete the missing Akkadian words masked by the [MASK] tokens and print out the restored document: {sample['Masked Document']}"},
        {"role": "assistant",
                   "content": f"{sample['Original Document']}"
                }]}
    return message


def replacenth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub.replace('\\',''), wanted, 1)
    newString = before + after
    return(newString)
def one_by_one(df, newest=False):
    new=[]
    for i,sample in df.iterrows():
        for word_i in range(sample['Masked Document'].count('[MASK]')):
            md=sample['Masked Document'].replace("[MASK]","[UNK]")
            md=replacenth(md,r"\[UNK\]",'[MASK]',word_i+1)
            conv=create_conversation_single(md,ast.literal_eval(sample['Masked Words'])[word_i], newest=args.newest_data)
            new.append(conv)
    return new

if args.one_by_one:
    import ast
    new_train=one_by_one(df_train, newest=args.newest_data)
    new_dev=one_by_one(df_dev, newest=args.newest_data)
    print(len(new_train))
    train_dataset = Dataset.from_list(new_train)
    dev_dataset = Dataset.from_list(new_dev)
    print(train_dataset)
else:
    train_dataset = Dataset.from_pandas(df_train)
    dev_dataset = Dataset.from_pandas(df_dev)
    if args.restore:
        train_dataset = train_dataset.map(create_conversation_restore, remove_columns=train_dataset.column_names)
        dev_dataset = dev_dataset.map(create_conversation_restore, remove_columns=dev_dataset.column_names)

    else:    
        train_dataset = train_dataset.map(create_conversation, remove_columns=train_dataset.column_names)
        dev_dataset = dev_dataset.map(create_conversation, remove_columns=dev_dataset.column_names)
    print(train_dataset)
print(train_dataset[0])
print(train_dataset[1])
print(train_dataset[2])
print(train_dataset[3])

print(train_dataset[6])
print(train_dataset[7])
print(train_dataset[8])
print(train_dataset[-1])



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from trl import setup_chat_format

# Hugging Face model id
model_id = "CohereForAI/c4ai-command-r-v01-4bit"
model_id = args.base_model

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
import os
if os.path.exists(args.save_dir):
   model=AutoPeftModelForCausalLM.from_pretrained(args.save_dir, device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        cache_dir="./models_cache")
   tokenizer = AutoTokenizer.from_pretrained(args.save_dir)

else:
# Load model and tokenizer
    if "command" in model_id:
        model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
        #cache_dir="./models_cache"
)
    else:
        model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
#        quantization_config=bnb_config
        #cache_dir="./models_cache"
)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
#tokenizer.padding_side = 'right' # to prevent warnings
print(tokenizer.apply_chat_template(train_dataset[0]["messages"], tokenize=False))
print(tokenizer.apply_chat_template(train_dataset[2]["messages"], tokenize=False))

print(tokenizer.vocab_size)
#print(tokenizer.get_vocab_size(with_added_tokens=True))
print(model.vocab_size)
#model, tokenizer = setup_chat_format(model, tokenizer)
from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=args.r//2,
        lora_dropout=0.05,
        r=args.r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)
batch=3
acc=12
if "EuroLLM" in model_id or "aya" in model_id:
    batch=4
    acc=10
if "Mistral" in model_id:
    batch=5
    acc=7
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir=args.save_dir, # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=batch,          # batch size per device during training
    gradient_accumulation_steps=acc,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_steps=150,                          # save checkpoint every 50 steps
    eval_steps=150,                         # evaluate every 100 steps
    save_strategy="steps",                  # save checkpoint every epoch
    learning_rate=args.lr,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

from trl import SFTTrainer

max_seq_length = 6144 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model
trainer.save_model()
