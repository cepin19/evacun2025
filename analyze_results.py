#!/usr/bin/env python
import argparse
import json
import collections
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the distribution of tokens (predicted and/or reference) in JSON files and visualize the relative frequencies."
    )
    parser.add_argument("input_files", nargs="+", help="Paths to input JSON files.")
    parser.add_argument("--field", type=str, default="Predicted",
                        help="Name of the field containing predicted tokens (default: Predicted). For aggregated files, you might use 'Top K Predictions'.")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top tokens to display (default: 20).")
    parser.add_argument("--mode_vis", type=str, choices=["predicted", "reference", "both"], default="predicted",
                        help="Select which tokens to visualize: 'predicted', 'reference', or 'both' (default: predicted).")
    parser.add_argument("--savefig", type=str, default=None,
                        help="If provided, save the figure to the given filename.")
    return parser.parse_args()

def main():
    args = parse_args()
    counter_pred = collections.Counter()
    counter_ref = collections.Counter()
    
    total_pred = 0
    total_ref = 0
    
    # Iterate over input files.
    for file in args.input_files:
        with open(file, "r") as f:
            data = json.load(f)
        # Use aggregated results if present.
        if "Aggregated Results" in data:
            entries = data["Aggregated Results"]
        else:
            entries = data
        
        for entry in entries:
            # For predicted tokens, use the field specified by --field.
            if args.mode_vis in ("predicted", "both"):
                pred_field = entry.get(args.field, [])
                # If field is a list of lists (e.g. top-k predictions), count each candidate.
                if isinstance(pred_field, list) and pred_field:
                    if isinstance(pred_field[0], list):
                        for sublist in pred_field:
                            if sublist:
                                for token in sublist:
                                    if token != "N/A":
                                        counter_pred[token] += 1
                                        total_pred += 1
                    else:
                        for token in pred_field:
                            if token != "N/A":
                                counter_pred[token] += 1
                                total_pred += 1
            # For reference tokens, use the "Reference" field.
            if args.mode_vis in ("reference", "both"):
                ref_field = entry.get("Reference", [])
                if isinstance(ref_field, list):
                    for token in ref_field:
                        counter_ref[token] += 1
                        total_ref += 1

    # Compute relative frequencies.
    # For predicted:
    pred_common = []
    if total_pred > 0:
        pred_common = [(token, count / total_pred) for token, count in counter_pred.most_common(args.top_n)]
    # For reference:
    ref_common = []
    if total_ref > 0:
        ref_common = [(token, count / total_ref) for token, count in counter_ref.most_common(args.top_n)]
    
    # Print results.
    if args.mode_vis in ("predicted", "both"):
        print("Top predicted tokens (relative frequencies):")
        for token, rel in pred_common:
            print(f"{token}: {rel:.4f} ({rel*100:.2f}%)")
    if args.mode_vis in ("reference", "both"):
        print("\nTop reference tokens (relative frequencies):")
        for token, rel in ref_common:
            print(f"{token}: {rel:.4f} ({rel*100:.2f}%)")
    
    # Visualization:
    plt.rcParams.update({'font.size': 35})
    if args.mode_vis == "predicted":
        tokens, freqs = zip(*pred_common) if pred_common else ([], [])
        plt.figure(figsize=(20, 12))
        plt.bar(tokens, freqs, color="skyblue")
        plt.xlabel("Predicted Token")
        plt.ylabel("Relative Frequency")
  #      plt.title("Top Predicted Tokens Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
    elif args.mode_vis == "reference":
        tokens, freqs = zip(*ref_common) if ref_common else ([], [])
        plt.figure(figsize=(20, 12))
        plt.bar(tokens, freqs, color="lightgreen")
        plt.xlabel("Reference Token")
        plt.ylabel("Relative Frequency")
   #     plt.title("Top Reference Tokens Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
    elif args.mode_vis == "both":
        fig, axs = plt.subplots(1, 2, figsize=(28, 12))
        if pred_common:
            tokens_pred, freqs_pred = zip(*pred_common)
            axs[0].bar(tokens_pred, freqs_pred, color="skyblue")
            axs[0].set_xlabel("Predicted Token")
            axs[0].set_ylabel("Relative Frequency")
    #        axs[0].set_title("Top Predicted Tokens")
            axs[0].tick_params(axis="x", rotation=45)
        else:
            axs[0].text(0.5, 0.5, "No predicted tokens", ha="center")
        if ref_common:
            tokens_ref, freqs_ref = zip(*ref_common)
            axs[1].bar(tokens_ref, freqs_ref, color="lightgreen")
            axs[1].set_xlabel("Reference Token")
            axs[1].set_ylabel("Relative Frequency")
     #       axs[1].set_title("Top Reference Tokens")
            axs[1].tick_params(axis="x", rotation=45)
        else:
            axs[1].text(0.5, 0.5, "No reference tokens", ha="center")
        plt.tight_layout()
    
    if args.savefig:
        plt.savefig(args.savefig,  bbox_inches='tight', pad_inches=0)
        print(f"Figure saved as {args.savefig}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

