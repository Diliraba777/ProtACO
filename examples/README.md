# Examples

This directory is reserved for small, non-sensitive examples that document the expected input formats. Do not place full research datasets, generated LMDB stores, or model checkpoints here.

Included files:

- `train_metadata.csv`: tiny training metadata example.
- `inference_truth.csv`: tiny inference truth table example.
- `sequences.fasta`: tiny FASTA example with matching IDs.

These files are format examples only. The pytest smoke tests create their own temporary random embeddings and do not require a real ProtT5 LMDB file.

## Training Metadata CSV

The training dataset currently expects a CSV file with at least these columns:

```csv
id,CO_score
protein_001,0.12
protein_002,0.35
protein_003,0.08
```

Each `id` must match a key in the LMDB embedding store.

`CO_score` is only an example target column name. Use `--target-col` to point the scripts to the ACO label column in your own metadata file.

## Inference Truth CSV

The inference script currently expects a truth CSV with at least these columns:

```csv
id,ACO_score
protein_001,0.12
protein_002,0.35
protein_003,0.08
```

`ACO_score` is only an example label column name. Use `--target-col` to point the inference script to the ACO label column in your own evaluation file.

## FASTA Input

The FASTA identifiers should match the `id` values in the CSV files:

```text
>protein_001
MKTAYIAKQRQISFVKSHFSRQDILD
>protein_002
GSHMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRF
```

## LMDB Embedding Store

The LMDB database should map each protein ID to a pickled NumPy array:

- protein-level embedding: shape `(D,)`
- token-level embedding: shape `(L, D)`

For ProtT5 token embeddings used by the current model configuration, `D` is expected to be `1024`.
