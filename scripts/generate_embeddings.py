#!/usr/bin/env python3
import argparse
import gc
import os
import pickle
from pathlib import Path

import lmdb
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ProtT5 embeddings and store them in LMDB.")
    parser.add_argument(
        "--model",
        default="Rostlab/prot_t5_xl_uniref50",
        help="Hugging Face model name or local path for ProtT5.",
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file.")
    parser.add_argument("--lmdb", required=True, help="Output LMDB path.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA to reduce memory usage.")
    parser.add_argument("--commit-interval", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=3500)
    parser.add_argument("--map-size-gb", type=int, default=3000, help="LMDB map size in GB.")
    parser.add_argument(
        "--trim-special-tokens",
        choices=["both", "end", "none"],
        default="both",
        help="How to trim special-token embeddings before saving. The legacy script used 'both'.",
    )
    return parser.parse_args()


def resolve_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_sequence(seq):
    return seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")


def trim_embeddings(hidden_states, mode):
    if mode == "both":
        return hidden_states[1:-1]
    if mode == "end":
        return hidden_states[:-1]
    return hidden_states


def load_existing_ids(lmdb_path):
    existing_ids = set()
    if not os.path.exists(lmdb_path):
        return existing_ids

    try:
        env_check = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env_check.begin() as txn:
            cursor = txn.cursor()
            for key, _ in tqdm(cursor, desc="Scanning DB", unit="keys"):
                existing_ids.add(key.decode())
        env_check.close()
        print(f"Found {len(existing_ids)} existing embeddings. These will be skipped.")
    except Exception as exc:
        print(f"Warning: Could not read existing DB ({exc}). Continuing in append mode.")
    return existing_ids


def embed_sequence(seq, tokenizer, model, device, trim_special_tokens):
    seq_spaced = " ".join(list(clean_sequence(seq)))
    tokenized = tokenizer(seq_spaced, return_tensors="pt", add_special_tokens=True)
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    with torch.no_grad():
        outputs = model(**tokenized)
        hidden_states = outputs.last_hidden_state.squeeze(0)

    hidden_states = trim_embeddings(hidden_states, trim_special_tokens)
    return hidden_states.cpu().numpy()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    lmdb_path = Path(args.lmdb)
    if lmdb_path.parent:
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading embedding model: {args.model}")
    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model)

    if args.fp16 and device.type == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()

    existing_ids = load_existing_ids(str(lmdb_path))

    todo_records = []
    print(f"Reading FASTA and filtering tasks: {args.fasta}")
    for record in tqdm(SeqIO.parse(args.fasta, "fasta"), desc="Filtering"):
        acc = record.id.split()[0]
        if acc in existing_ids:
            continue
        if len(record.seq) > args.max_seq_len:
            continue
        todo_records.append(record)

    todo_records.sort(key=lambda record: len(record.seq))
    print(f"Remaining sequences to process: {len(todo_records)}")

    if not todo_records:
        print("All sequences have already been processed or were filtered out.")
        return

    map_size = args.map_size_gb * 1024**3
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    txn = env.begin(write=True)
    count = 0
    batch_counter = 0

    try:
        for record in tqdm(todo_records, desc="Processing"):
            acc = record.id.split()[0]
            seq_str = str(record.seq)

            try:
                emb_numpy = embed_sequence(
                    seq_str,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    trim_special_tokens=args.trim_special_tokens,
                )
                txn.put(acc.encode(), pickle.dumps(emb_numpy))
                count += 1
                batch_counter += 1

                if batch_counter >= args.commit_interval:
                    txn.commit()
                    txn = env.begin(write=True)
                    batch_counter = 0
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"\nCUDA OOM on {acc} (len={len(seq_str)}). Skipping.")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(f"\nRuntimeError on {acc}: {exc}")
            except Exception as exc:
                if "Input/output error" in str(exc) or "No space left" in str(exc):
                    print("\nCritical disk or IO error. Committing current batch before exiting.")
                    raise
                print(f"\nError on {acc}: {exc}")

        txn.commit()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Committing current batch.")
        txn.commit()
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        try:
            txn.commit()
        except Exception:
            pass
        raise
    finally:
        env.sync()
        env.close()

    print("-" * 50)
    print("Job finished.")
    print(f"Newly added: {count}")
    print(f"Database: {lmdb_path}")


if __name__ == "__main__":
    main()
