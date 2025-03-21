#!/usr/bin/env python
import argparse
import json
import collections

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count how many tokens in a specified field have relative frequency above a defined threshold."
    )
    parser.add_argument("input_files", nargs="+", help="Paths to input JSON files.")
    parser.add_argument("--field", type=str, default="Predicted",
                        help="Field to analyze (default: Predicted). For aggregated files, you might use 'Top K Predictions'.")
    parser.add_argument("--threshold", type=float, default=0.0001,
                        help="Relative frequency threshold (default: 0.0001).")
    return parser.parse_args()

def main():
    args = parse_args()
    counter = collections.Counter()
    total_count = 0

    # Iterate over all input files.
    for file in args.input_files:
        with open(file, "r") as f:
            data = json.load(f)
        # Use "Aggregated Results" if present.
        if "Aggregated Results" in data:
            entries = data["Aggregated Results"]
        else:
            entries = data

        for entry in entries:
            field_val = entry.get(args.field, [])
            # If the field is a list of lists (e.g., top-k predictions), flatten it.
            if isinstance(field_val, list) and field_val:
                if isinstance(field_val[0], list):
                    for sublist in field_val:
                        for token in sublist:
                            if token != "N/A":
                                counter[token] += 1
                                total_count += 1
                else:
                    for token in field_val:
                        if token != "N/A":
                            counter[token] += 1
                            total_count += 1

    if total_count == 0:
        print("No tokens found in the specified field.")
        return

    # Compute relative frequencies and count tokens above the threshold.
    tokens_above_threshold = []
    for token, count in counter.items():
        rel_freq = count / total_count
        if rel_freq > args.threshold:
            tokens_above_threshold.append((token, rel_freq))
    
    tokens_above_threshold.sort(key=lambda x: x[1], reverse=True)
    print(f"Total tokens counted: {total_count}")
        
    unique_tokens = len(counter)
    
    print(f"Number of unique tokens: {unique_tokens}")
    print(f"Number of tokens with relative frequency above {args.threshold}: {len(tokens_above_threshold)}\n")
    #print("Tokens above threshold (token: relative frequency):")
    for token, rel_freq in tokens_above_threshold:
        pass
        #print(f"{token}: {rel_freq:.4f} ({rel_freq*100:.2f}%)")

if __name__ == "__main__":
    main()

