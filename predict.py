#!/usr/bin/env python
import argparse
import sys
import pandas as pd
import ast
import torch
import re
import json
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM client for Akkadian prompt completion with multiple modes, accuracy evaluation, and JSON output."
    )
    parser.add_argument("--model", required=True,
                        help="Path to the base model.")
    parser.add_argument("--peft_model", default=None,
                        help="Path to the PEFT model (optional).")
    parser.add_argument("--hf_token",
                        help="Hugging Face authentication token.")
    parser.add_argument("--mode", choices=["all", "onebyone","default", "single", "restore"], default="all",
                        help=("Select the prompt mode. 'default' creates a prompt with all missing words (word-by-word accuracy), "
                              "'single' creates one prompt per masked word, and "
                              "'restore' produces a full restored document and evaluates each mask position."))
    parser.add_argument("--input", type=argparse.FileType('r'), default=sys.stdin,
                        help="Input CSV file (default: stdin).")
    parser.add_argument("--max_new_tokens", type=int, default=900,
                        help="Maximum number of new tokens to generate (default: 600).")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for generation (default: 0.2).")
    parser.add_argument("--do_sample", type=bool, default=True,
                        help="Sample?")

    parser.add_argument("--output_json", type=str, default="results.json",
                        help="Output JSON file to save predictions (default: results.json).")
    return parser.parse_args()

def normalize(text):
    """Normalize text for comparison (lowercase and strip whitespace)."""
    return text.strip().lower()

def replacenth(string, sub, wanted, n):
    """
    Replaces the nth occurrence of substring 'sub' in 'string' with 'wanted'.
    The 'sub' argument is provided as a regex pattern (e.g. r"\[UNK\]").
    """
    matches = [m.start() for m in re.finditer(sub, string)]
    if len(matches) < n:
        return string
    where = matches[n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub.replace('\\', ''), wanted, 1)
    return before + after

def get_prompt(sample, mode):
    """
    Build the conversation prompt for modes 'default' and 'restore'.
    The sample is assumed to be a dict-like row with keys:
      - "Masked Document"
      - "Masked Words" (for default)
      - "Original Document" (for restore)
    """
    if mode == "all" or mode == "default":
        prompt_text = (
            f"Fill in the missing Akkadian words, masked by the [MASK] token. "
            f"Output \"WORDS:\" and a comma-separated list of the missing words in original Akkadian: "
            f"{sample['Masked Document']}"
        )
        return [{"role": "user", "content": prompt_text}]
    elif mode == "restore":
        prompt_text = (
            f"Complete the missing Akkadian words masked by the [MASK] tokens and print out the restored document: "
            f"{sample['Masked Document']}"
        )
        return [{"role": "user", "content": prompt_text}]
    else:
        raise ValueError("get_prompt is not intended for 'single' mode.")

def complete_restore(sample, do_sample=True):
    prompt = get_prompt(sample, "restore")
    print(sample['Masked Document'], file=sys.stderr)
    forced_parts = sample['Masked Document'].split("[MASK]")
    input_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    # Process each forced part sequentially.
    for i, forced_part in enumerate(forced_parts):
        if forced_part.rstrip() != '':
            forced_part = forced_part.rstrip()
        if forced_part.strip() != '':
            forced_ids = tokenizer.encode(forced_part, add_special_tokens=False, return_tensors="pt").to(device)
            input_ids = torch.cat([input_ids, forced_ids], dim=-1)
        len_before = len(tokenizer.decode(input_ids[0]).split())
        with torch.no_grad():
            outputs = peft_model(input_ids, use_cache=True)
            past = outputs.past_key_values  # Cache the past key values.
        # If this is not the last forced part (or the forced part is just a space), generate a word.
        if i != (len(forced_parts) - 1) or forced_part == " ":
            with torch.no_grad():
                first_word = True
                for subword_i in range(20):
                    last_token = input_ids[:, -1:]
                    outputs = peft_model(last_token, past_key_values=past, use_cache=True)
                    logits = outputs.logits
                    # Get the candidate token via argmax.
                    candidate_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                    candidate_token_str = tokenizer.convert_ids_to_tokens(candidate_token_id[0])
                    # If the candidate token starts with the new-word marker (and isn’t empty) or we’re in a later iteration, accept it.
                    if (candidate_token_str[0].startswith("Ġ") or subword_i > 0) and tokenizer.decode(candidate_token_id[0]).strip() != "":
                        next_token_id = candidate_token_id
                    else:
                        # Otherwise, sample tokens until one starts with "Ġ" and is non-empty.
                        next_token_id = tokenizer.encode(" none", add_special_tokens=False, return_tensors="pt").to(device)
                        probs = torch.softmax(logits[:, -1, :], dim=-1)
                        for attempt in range(200000):
                            sampled_token_id = torch.multinomial(probs, num_samples=1)
                            sampled_token_str = tokenizer.convert_ids_to_tokens(sampled_token_id[0])
                            if sampled_token_str[0].startswith("Ġ") and tokenizer.decode(sampled_token_id[0]).strip() != "":
                                next_token_id = sampled_token_id
                                break
                    token_str = tokenizer.convert_ids_to_tokens(next_token_id.squeeze().tolist())
                    # If the new token starts with "Ġ" and this is not the first subword, break the loop.
                    if token_str[0].startswith("Ġ") and not first_word:
                        break
                    # If we hit a termination token, break.
                    if tokenizer.decode(next_token_id[0]) in ["<BOS_TOKEN>", "<EOS_TOKEN>", "<|END_OF_TURN_TOKEN|>", "<|CHATBOT_TOKEN|>"]:
                        break
                    past = outputs.past_key_values
                    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                    first_word = False
        # Verify that exactly one token was added.
        if (len(tokenizer.decode(input_ids[0]).split()) != (len_before + 1)) and (i != (len(forced_parts) - 1)) and (i != 0):
            print(i, file=sys.stderr)
            print(len(forced_parts) - 1, file=sys.stderr)
            print(len(tokenizer.decode(input_ids[0]).split()), file=sys.stderr)
            print(len_before, file=sys.stderr)
            print("not added one token!", file=sys.stderr)
            print(forced_part, file=sys.stderr)
            print(tokenizer.decode(input_ids[0]), file=sys.stderr)
    print(f"all {tokenizer.decode(input_ids[0])}", file=sys.stderr)
    decoded = tokenizer.decode(input_ids[0])
    if "<|CHATBOT_TOKEN|>" in decoded:
        completion = decoded.split("<|CHATBOT_TOKEN|>")[-1]
    elif "[/INST]" in decoded:
        completion=decoded.split("[/INST]")[-1].replace('</s>',"")
    else:
        completion = decoded
    return completion.replace('<|END_OF_TURN_TOKEN|>', '').strip()

def complete(sample, mode, max_new_tokens, temperature, do_sample=True):
    """
    For default (and fallback restore) modes: build a prompt from the sample and generate a completion.
    """
    conversation = get_prompt(sample, mode)
    print("Using conversation prompt:", conversation, file=sys.stderr)
    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    gen_tokens = peft_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    decoded = tokenizer.decode(gen_tokens[0])
    if "<|CHATBOT_TOKEN|>" in decoded:
        completion = decoded.split("<|CHATBOT_TOKEN|>")[-1]
    elif "[/INST]" in decoded:
        completion=decoded.split("[/INST]")[-1].replace('</s>',"")
    else:
        completion = decoded
    return completion.replace('<|END_OF_TURN_TOKEN|>', '').strip()

def complete_one_by_one(sample, max_new_tokens, temperature, do_sample=True):
    """
    For 'single' mode: for each occurrence of [MASK] in sample['Masked Document'],
    create a prompt where all masked tokens are replaced by [UNK] except one selected [MASK].
    Returns a list of tuples: (modified_document, generated_output, reference_word)
    """
    md_orig = sample["Masked Document"]
    n_masks = md_orig.count("[MASK]")
    completions = []
    try:
        ref_words = ast.literal_eval(sample["Masked Words"])
    except Exception as e:
        print(f"Warning: Could not parse Masked Words for sample: {e}", file=sys.stderr)
        ref_words = sample["Masked Words"]
    for i in range(n_masks):
        md_all_unk = md_orig.replace("[MASK]", "[UNK]")
        md_modified = replacenth(md_all_unk, r"\[UNK\]", "[MASK]", i+1)
        prompt_text = f"Fill in the missing Akkadian word masked by the [MASK] token: {md_modified}"
        conversation = [{"role": "user", "content": prompt_text}]
        print(f"Using conversation prompt (mask #{i+1}):", conversation, file=sys.stderr)
        input_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        gen_tokens = peft_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        decoded = tokenizer.decode(gen_tokens[0])
        if "<|CHATBOT_TOKEN|>" in decoded:
            output = decoded.split("<|CHATBOT_TOKEN|>")[-1]
        elif "[/INST]" in decoded:
            output=decoded.split("[/INST]")[-1].replace('</s>',"")
        else:
            output = decoded
        output = output.replace('<|END_OF_TURN_TOKEN|>', '').strip()
        reference = ref_words[i] if isinstance(ref_words, list) and len(ref_words) > i else ref_words
        completions.append((md_modified, output, reference))
    return completions

def evaluate_restore(masked_doc, original_doc, generated_doc):
    """
    Tokenize the masked document, original document, and generated document (using whitespace splitting).
    Identify positions of the [MASK] tokens in the masked document.  
    If the generated document does not have the same number of tokens as the masked document,
    return predicted tokens as "N/A" for each mask position.
    
    Returns a tuple: (num_correct, num_masks, predicted_list, reference_list)
    """
    tokens_masked = masked_doc.split()
    tokens_generated = generated_doc.split()
    tokens_original = original_doc.split()
    mask_positions = [i for i, token in enumerate(tokens_masked) if token.strip() == "[MASK]"]
    print(f"lenghts Generated: {len(tokens_generated)} Masked: {len(tokens_masked)}")
    if len(tokens_generated) != len(tokens_masked):
        print(f"Incorrect lenghts Generated: {len(tokens_generated)} Masked: {len(tokens_masked)} ")
        predicted_list = ["N/A" for _ in mask_positions]
        ref_list = [tokens_original[i] if i < len(tokens_original) else "N/A" for i in mask_positions]
        return 0, len(mask_positions), predicted_list, ref_list
    correct = 0
    predicted_list = []
    for pos in mask_positions:
        pred_tok = tokens_generated[pos] if pos < len(tokens_generated) else "N/A"
        predicted_list.append(pred_tok)
        ref_tok = tokens_original[pos] if pos < len(tokens_original) else "N/A"
        if normalize(pred_tok) == normalize(ref_tok):
            correct += 1
    ref_list = [tokens_original[i] if i < len(tokens_original) else "N/A" for i in mask_positions]
    return correct, len(mask_positions), predicted_list, ref_list

# --- Main script begins here ---
if __name__ == "__main__":
    args = parse_args()

    print("Loading base model from:", args.model, file=sys.stderr)
    peft_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.peft_model:
        print("Loading PEFT model from:", args.peft_model, file=sys.stderr)
        peft_model = PeftModel.from_pretrained(
            peft_model,
            args.peft_model,
            torch_dtype=torch.bfloat16,
            is_trainable=False
        )

    col_names = ["Document ID", "Copy ID", "Dataset Type",
                 "Original Document", "Masked Document", "Masked Words"]
    docs = pd.read_csv(args.input, skiprows=1, dtype=str, names=col_names)

    results = []  # Will hold one dict per document with required fields.

    # Global counters (for logging purposes)
    overall_total_predictions = 0
    overall_correct_predictions = 0

    for idx, row in docs.iterrows():
        print(f"Processing document {row['Document ID']}...", file=sys.stderr)
        # Initialize per-document counters (if desired)
        total_predictions = 0
        correct_predictions = 0

        # This dictionary will hold our saved output.
        output_entry = {
            "Document ID": row["Document ID"],
            "Copy ID": row["Copy ID"],
            "Original Document": row["Original Document"],
            "Masked Document": row["Masked Document"],
            "Reference": None,  # To be filled as a list
            "Predicted": None   # To be filled as a list
        }

        if args.mode == "single" or args.mode=="onebyone":
            completions = complete_one_by_one(row, args.max_new_tokens, args.temperature, do_sample=args.do_sample)
            ref_list = []
            pred_list = []
            for i, (modified_doc, output, reference) in enumerate(completions, start=1):
                print(f"Document {row['Document ID']} - Mask #{i}:", file=sys.stderr)
                print("Modified Document:", modified_doc, file=sys.stderr)
                print("Output:", output, file=sys.stderr)
                print("Reference:", reference, file=sys.stderr)
                total_predictions += 1
                if normalize(output) == normalize(str(reference)):
                    correct_predictions += 1
                ref_list.append(str(reference))
                pred_list.append(output)
            output_entry["Reference"] = ref_list
            output_entry["Predicted"] = pred_list
            overall_total_predictions += total_predictions
            overall_correct_predictions += correct_predictions

        elif args.mode == "default" or args.mode=="all":
            output_text = complete(row, args.mode, args.max_new_tokens, args.temperature, do_sample=args.do_sample)
            # Remove the "WORDS:" prefix if present.
            predicted_text=output_text.split("WORDS:")[-1].replace('</s>','').strip()
            #if output_text.lower().startswith("words:"):
            #predicted_text = output_text[len("words:"):].strip()
            #else:
             #   predicted_text = output_text.strip()
            # Try to parse the predicted words as a list; if that fails, split on commas.
            try:
                predicted_words = ast.literal_eval(predicted_text)
                if isinstance(predicted_words, str):
                    predicted_words = [w.strip() for w in predicted_words.split(",") if w.strip()]
            except Exception:
                predicted_words = [w.strip() for w in predicted_text.split(",") if w.strip()]
            try:
                reference_words = ast.literal_eval(row["Masked Words"])
                if isinstance(reference_words, str):
                    reference_words = [w.strip() for w in reference_words.split(",") if w.strip()]
            except Exception:
                reference_words = [w.strip() for w in row["Masked Words"].split(",") if w.strip()]
    # --- NEW CODE: Force the number of predicted words to match the reference ---
            target_len = len(reference_words)
            if len(predicted_words) > target_len:
                predicted_words = predicted_words[:target_len]
            elif len(predicted_words) < target_len:
        # Here we pad with empty strings; you could also use a placeholder like "N/A"
                predicted_words = predicted_words + [""] * (target_len - len(predicted_words))
    # --- END NEW CODE ---


            sample_total = len(reference_words)
            sample_correct = 0
            for pred, ref in zip(predicted_words, reference_words):
                if normalize(pred) == normalize(ref):
                    sample_correct += 1
            total_predictions += sample_total
            correct_predictions += sample_correct
            print(f"Document {row['Document ID']} Predicted words:", predicted_words, file=sys.stderr)
            print("Reference words:", reference_words, file=sys.stderr)
            output_entry["Reference"] = reference_words
            output_entry["Predicted"] = predicted_words
            overall_total_predictions += total_predictions
            overall_correct_predictions += correct_predictions

        elif args.mode == "restore":
            # Use the complete_restore function to generate a restored document.
            output_text = complete_restore(row, do_sample=args.do_sample)
            # Evaluate token-level predictions at mask positions.
            correct, sample_total, predicted_list, ref_list = evaluate_restore(
                row["Masked Document"],
                row["Original Document"],
                output_text
            )
            total_predictions += sample_total
            correct_predictions += correct
            print(f"Document {row['Document ID']} Output:", output_text, file=sys.stderr)
            print("Reference Original Document:", row["Original Document"], file=sys.stderr)
            print(f"Mask positions evaluated: {sample_total}, Correct: {correct}", file=sys.stderr)
            output_entry["Reference"] = ref_list
            output_entry["Predicted"] = predicted_list
            overall_total_predictions += total_predictions
            overall_correct_predictions += correct

        else:
            print(f"Unsupported mode {args.mode}", file=sys.stderr)
            exit()

        results.append(output_entry)
        print("-----", file=sys.stderr)

    if overall_total_predictions > 0:
        overall_accuracy = overall_correct_predictions / overall_total_predictions * 100
        print(f"Overall Accuracy: {overall_correct_predictions}/{overall_total_predictions} ({overall_accuracy:.2f}%)", file=sys.stderr)
    else:
        print("No predictions were made.", file=sys.stderr)

    # Save the results to a JSON file.
    with open(args.output_json, "w") as out_f:
        json.dump(results, out_f, indent=2)
    print(f"Saved predictions to {args.output_json}", file=sys.stderr)

