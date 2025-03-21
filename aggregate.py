#!/usr/bin/env python
import argparse
import json
import collections

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple JSON prediction files and output the top k predicted tokens for each mask position along with accuracy statistics."
    )
    parser.add_argument("input_files", nargs="+",
                        help="Paths to input JSON files.")
    parser.add_argument("--output", type=str, default="aggregated.json",
                        help="Output JSON file (default: aggregated.json).")
    parser.add_argument("--best_n", type=int, default=None,
                        help="If provided, only the best n files (by accuracy) are used for aggregation.")
    parser.add_argument("--test", action="store_true",
                        help="Run the script in test mode (aggregate predictions only, without using references for accuracy).")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Output top k predicted words and compute top k accuracy (default: 1).")
    return parser.parse_args()

def main():
    args = parse_args()

    # First pass: read all files and (if not in test mode) compute per-file accuracy.
    file_data = {}
    file_accuracies = {}
    total_file_correct = 0
    total_file_total = 0

    for file in args.input_files:
        with open(file, "r") as f:
            data = json.load(f)
        file_data[file] = data
        if not args.test:
            file_correct = 0
            file_total = 0
            for entry in data:
                preds = entry.get("Predicted", [])
                ref = entry.get("Reference", [])
                if ref and (len(preds) == len(ref)):
                    for p, r in zip(preds, ref):
                        if p == r:
                            file_correct += 1
                    file_total += len(ref)
            accuracy = (file_correct / file_total * 100) if file_total > 0 else 0
            file_accuracies[file] = {"correct": file_correct, "total": file_total, "accuracy": accuracy}
            print(f"File: {file} -- Accuracy: {accuracy:.2f}% ({file_correct}/{file_total})")
            total_file_correct += file_correct
            total_file_total += file_total
        else:
            file_accuracies[file] = {"correct": None, "total": None, "accuracy": None}
            print(f"File: {file} -- Test mode (no accuracy computed)")

    if not args.test:
        overall_file_accuracy = (total_file_correct / total_file_total * 100) if total_file_total > 0 else 0
    else:
        overall_file_accuracy = "N/A"

    # Sorting: If not test mode, sort by accuracy (highest first); otherwise, sort alphabetically.
    if not args.test:
        sorted_files = sorted(file_data.keys(), key=lambda f: file_accuracies[f]['accuracy'], reverse=True)
    else:
        sorted_files = sorted(file_data.keys())
    print("\nFiles sorted by accuracy:" if not args.test else "\nFiles (sorted alphabetically):")
    for file in sorted_files:
        acc = file_accuracies[file]['accuracy']
        if acc is not None:
            print(f"{file}: {acc:.2f}%")
        else:
            print(file)

    # If best_n is provided, select only the top best_n files.
    if args.best_n is not None:
        best_files = sorted_files[:args.best_n]
        print(f"\nUsing only the best {args.best_n} files for aggregation:")
        for file in best_files:
            if file_accuracies[file]['accuracy'] is not None:
                print(f"{file}: {file_accuracies[file]['accuracy']:.2f}%")
            else:
                print(file)
    else:
        best_files = sorted_files

    # Second pass: aggregate predictions from the selected files.
    # We use a composite key (Document ID, Copy ID) to group entries.
    aggregated = {}
    for file in best_files:
        data = file_data[file]
        for entry in data:
            key = (entry["Document ID"], entry["Copy ID"])
            if key not in aggregated:
                aggregated[key] = []
            # Store a tuple of (Predicted, Reference, Masked Document).
            aggregated[key].append((
                entry.get("Predicted", []),
                entry.get("Reference", []),
                entry.get("Masked Document", "")
            ))

    aggregated_results = []
    overall_correct_agg = 0
    overall_total_agg = 0

    for key, entries in aggregated.items():
        if not entries:
            continue
        # Use the first entry's masked document as baseline.
        masked_doc0 = entries[0][2]
        expected_count = masked_doc0.count("[MASK]")
        # Verify that all entries have the same number of [MASK] tokens.
        for _, _, mdoc in entries:
            if mdoc.count("[MASK]") != expected_count:
                print(f"Warning: Inconsistent mask counts for document {key}: expected {expected_count}, got {mdoc.count('[MASK]')}", file=sys.stderr)
                break

        top_k_all = []
        # For each mask position from 0 to expected_count-1, aggregate predictions.
        for pos in range(expected_count):
            counter = collections.Counter()
            for (preds, _, _) in entries:
                if pos < len(preds):
                    counter[preds[pos]] += 1
                else:
                    counter["N/A"] += 1
            # Get all candidates sorted by count.
            sorted_items = counter.most_common()
            # If there is any candidate other than "N/A", filter out "N/A".
            non_na_items = [ (token, count) for token, count in sorted_items if token != "N/A" ]
            if non_na_items:
                # Use the top_k from the non-"N/A" candidates.
                top_k_tokens = [token for token, count in non_na_items[:args.top_k]]
            else:
                top_k_tokens = ["N/A"] * args.top_k
            top_k_all.append(top_k_tokens)

        if not args.test:
            ref = entries[0][1]
            # Adjust the reference list to match expected_count.
            if len(ref) > expected_count:
                ref = ref[:expected_count]
            elif len(ref) < expected_count:
                ref = ref + ["N/A"] * (expected_count - len(ref))
            doc_correct = sum(1 for pos, ref_tok in enumerate(ref) if pos < len(top_k_all) and ref_tok in top_k_all[pos])
            doc_total = len(ref)
            doc_accuracy = (doc_correct / doc_total * 100) if doc_total > 0 else 0
            overall_correct_agg += doc_correct
            overall_total_agg += doc_total
            aggregated_results.append({
                "Document ID": key[0],
                "Copy ID": key[1],
                "Top K Predictions": top_k_all,
                "Reference": ref,
                "Document Top-K Accuracy": doc_accuracy,
                "Correct": doc_correct,
                "Total": doc_total,
                "Num Runs": len(entries)
            })
        else:
            aggregated_results.append({
                "Document ID": key[0],
                "Copy ID": key[1],
                "Top K Predictions": top_k_all,
                "Reference": None,
                "Document Top-K Accuracy": "N/A",
                "Correct": "N/A",
                "Total": "N/A",
                "Num Runs": len(entries)
            })

    if not args.test and overall_total_agg > 0:
        overall_agg_accuracy = (overall_correct_agg / overall_total_agg * 100)
    else:
        overall_agg_accuracy = "N/A"

    aggregated_summary = {
        "Per-File Accuracies": file_accuracies,
        "Overall File Accuracy": overall_file_accuracy,
        "Overall Aggregated Top-K Accuracy": overall_agg_accuracy,
        "Overall Aggregated Correct": overall_correct_agg if not args.test else "N/A",
        "Overall Aggregated Total": overall_total_agg if not args.test else "N/A",
        "Aggregated Results": aggregated_results
    }

    with open(args.output, "w") as out_f:
        json.dump(aggregated_summary, out_f, indent=2)
    if not args.test:
        print(f"\nFinal Overall File Accuracy: {overall_file_accuracy:.2f}% ({total_file_correct}/{total_file_total})")
        print(f"Final Overall Aggregated Top-K Accuracy: {overall_agg_accuracy:.2f}% ({overall_correct_agg}/{overall_total_agg})")
    else:
        print("\nTest mode enabled; accuracy metrics are not computed.")
    print(f"Aggregated results saved to {args.output}")

if __name__ == "__main__":
    main()

