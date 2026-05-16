import argparse
import os
import re
from pathlib import Path

import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, T5Tokenizer

from protaco.model import TransformerEncoderRegressor


class FastaInferenceDataset(Dataset):
    def __init__(self, fasta_path, truth_csv=None, id_column="id", target_column="ACO_score"):
        self.id_to_score = None
        if truth_csv:
            truth_df = pd.read_csv(truth_csv)
            missing_columns = [col for col in (id_column, target_column) if col not in truth_df.columns]
            if missing_columns:
                raise ValueError(
                    f"Missing required column(s) in {truth_csv}: {', '.join(missing_columns)}"
                )
            truth_df[id_column] = truth_df[id_column].astype(str)
            self.id_to_score = dict(zip(truth_df[id_column], truth_df[target_column]))

        self.data = []
        print(f"Reading FASTA: {fasta_path}")
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id = str(record.id)
            if self.id_to_score is not None and seq_id not in self.id_to_score:
                continue

            seq = str(record.seq)
            formatted_seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
            item = {
                "id": seq_id,
                "seq": formatted_seq,
                "len": len(seq),
            }
            if self.id_to_score is not None:
                item["y_true"] = self.id_to_score[seq_id]
            self.data.append(item)

        print(f"Loaded {len(self.data)} sequences.")
        if not self.data:
            raise ValueError("No FASTA records were loaded. Check IDs and input paths.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ProtACO inference on FASTA sequences.")
    parser.add_argument("--checkpoint", required=True, help="Trained Lightning checkpoint path.")
    parser.add_argument("--fasta", required=True, help="Input FASTA file.")
    parser.add_argument("--output-csv", required=True, help="Output CSV for predictions.")
    parser.add_argument("--truth-csv", default=None, help="Optional truth CSV for reporting y_true.")
    parser.add_argument("--id-col", default="id", help="ID column in the truth CSV.")
    parser.add_argument("--target-col", default="ACO_score", help="Target column in the truth CSV.")
    parser.add_argument(
        "--model",
        default="Rostlab/prot_t5_xl_uniref50",
        help="Hugging Face model name or local path for ProtT5.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(args):
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading embedding model: {args.model}")

    try:
        tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=False)
        embed_model = T5EncoderModel.from_pretrained(args.model).to(device)
        embed_model.eval()
    except Exception as exc:
        print("Error loading ProtT5. Make sure transformers and sentencepiece are installed.")
        raise exc

    print(f"Loading trained regressor: {args.checkpoint}")
    model = TransformerEncoderRegressor.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    dataset = FastaInferenceDataset(
        args.fasta,
        truth_csv=args.truth_csv,
        id_column=args.id_col,
        target_column=args.target_col,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    results = []
    print("Starting inference loop.")
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["id"]
            seqs = batch["seq"]

            tokenized = tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            ).to(device)

            attention_mask = tokenized["attention_mask"]
            embedding_output = embed_model(
                input_ids=tokenized["input_ids"],
                attention_mask=attention_mask,
            )
            embeddings = embedding_output.last_hidden_state
            model_mask = attention_mask == 0
            embeddings = embeddings * attention_mask.unsqueeze(-1)

            preds = model(embeddings, mask=model_mask).cpu().numpy()

            y_true = batch.get("y_true")
            for idx, seq_id in enumerate(ids):
                row = {
                    "id": seq_id,
                    "y_pred": float(preds[idx]),
                }
                if y_true is not None:
                    row["y_true"] = float(y_true[idx])
                results.append(row)

            print(f"Processed {len(results)} / {len(dataset)}", end="\r")

    output_path = Path(args.output_csv)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_res = pd.DataFrame(results)
    ordered_columns = ["id", "y_pred"] + (["y_true"] if "y_true" in df_res.columns else [])
    df_res = df_res[ordered_columns]
    df_res.to_csv(output_path, index=False)

    print(f"\nDone. Results saved to {output_path}")
    print("Preview:")
    print(df_res.head())


def main():
    args = parse_args()
    for path_name in ("checkpoint", "fasta"):
        path = getattr(args, path_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} not found: {path}")
    if args.truth_csv and not os.path.exists(args.truth_csv):
        raise FileNotFoundError(f"truth_csv not found: {args.truth_csv}")
    run_inference(args)


if __name__ == "__main__":
    main()
