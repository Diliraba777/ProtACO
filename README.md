# ProtACO

ProtACO is a PyTorch Lightning implementation for predicting protein absolute contact order (ACO) from sequence-derived protein language model representations. The workflow uses ProtT5 embeddings, a Transformer encoder regressor, and an MSLE plus pairwise-difference training objective.

This repository accompanies the manuscript **"ProtACO: Prediction of Protein Absolute Contact Order from Sequence Using Pre-trained Language Models"**. It provides the model implementation, data-loading utilities, embedding generation script, training script, inference script, input-format examples, and smoke tests with synthetic embeddings.

## Project Layout

```text
.
|-- protaco/                 # Importable package with dataset and model code
|   |-- __init__.py
|   |-- dataset.py           # Dataset and batch collation for protein embeddings
|   `-- model.py             # Transformer encoder regressor
|-- scripts/                 # Command-line entry points
|   |-- generate_embeddings.py
|   |-- inference.py
|   `-- train.py
|-- examples/                # Data format notes and tiny example files
|-- tests/                   # Pytest smoke tests with synthetic embeddings
|-- .github/workflows/       # GitHub Actions smoke-test workflow
|-- CITATION.cff             # Citation metadata
|-- pyproject.toml           # Packaging and test metadata
|-- requirements.txt         # Python dependencies
|-- LICENSE                  # MIT license
|-- THIRD_PARTY_NOTICES.md   # Notices for adapted third-party work
`-- .gitignore               # Files excluded from version control
```

## Installation

Create a clean Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

For development or testing, install the repository in editable mode:

```bash
pip install -e ".[dev]"
```

PyTorch installation can depend on the local CUDA version. If the command above installs a CPU-only build or an incompatible CUDA build, install PyTorch first using the command recommended by the official PyTorch selector, then install the remaining dependencies.

## Data Format

Training expects:

- A CSV metadata file containing an `id` column.
- A numeric target column containing the ACO value.
- An LMDB database where each key is the same protein `id`, and each value is a pickled NumPy array containing either a protein-level embedding with shape `(D,)` or token-level embeddings with shape `(L, D)`.

Inference expects:

- A FASTA file containing protein sequences.
- A trained PyTorch Lightning checkpoint passed through `--checkpoint`.
- Optionally, a truth CSV containing sequence IDs and ACO labels for evaluation.

The example files use `CO_score` and `ACO_score` as simple CSV column names. These are implementation-level placeholders rather than manuscript terminology. Use `--id-col` and `--target-col` to point the scripts to the actual column names in local metadata files.

Small example CSV and FASTA files are included in `examples/` to document the expected input formats. They are not intended to reproduce manuscript results.

## Usage

### Generate ProtT5 Embeddings

```bash
python -m scripts.generate_embeddings \
  --model Rostlab/prot_t5_xl_uniref50 \
  --fasta examples/sequences.fasta \
  --lmdb outputs/prottrans_embeddings.lmdb \
  --device auto \
  --fp16
```

This command reads protein sequences from FASTA, extracts ProtT5 embeddings, and stores them in an LMDB database. For local experiments, provide the path to the sequence file to be processed.

### Train ProtACO

```bash
python -m scripts.train \
  --csv examples/train_metadata.csv \
  --lmdb outputs/prottrans_embeddings.lmdb \
  --target-col CO_score \
  --output-dir runs/protaco \
  --accelerator auto \
  --devices auto
```

For multi-GPU training, pass explicit PyTorch Lightning options:

```bash
python -m scripts.train \
  --csv path/to/train_metadata.csv \
  --lmdb path/to/prottrans_embeddings.lmdb \
  --target-col ACO_column_name \
  --output-dir runs/protaco \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp
```

Training outputs are written under the selected `--output-dir`, including logs, checkpoints, validation predictions, and plots.

### Run Inference

```bash
python -m scripts.inference \
  --checkpoint path/to/best.ckpt \
  --fasta examples/sequences.fasta \
  --truth-csv examples/inference_truth.csv \
  --target-col ACO_score \
  --output-csv outputs/predictions.csv \
  --device auto
```

Inference requires a trained PyTorch Lightning checkpoint. The checkpoint used for the manuscript is not included in this repository. Users can provide their own checkpoint or retrain the model with `python -m scripts.train`.

Use `python -m scripts.train --help`, `python -m scripts.generate_embeddings --help`, or `python -m scripts.inference --help` to see all available options, including batch size, worker count, model dimensions, split ratio, GPU settings, and custom CSV column names.

## Smoke Tests

The test suite uses synthetic random embeddings and a temporary LMDB database, so it does not require the original dataset, trained checkpoints, or a downloaded ProtT5 model.

Run the tests with:

```bash
pytest
```

The tests cover:

- LMDB-backed dataset loading
- variable-length embedding padding
- missing CSV column errors
- model forward pass
- training-step loss calculation
- pairwise loss behavior

GitHub Actions is configured in `.github/workflows/tests.yml` to run these smoke tests on pushes and pull requests.

## Data and Model Availability

The training data, FASTA sequences used in the manuscript, generated ProtT5 LMDB embeddings, trained checkpoint, and manuscript prediction outputs are not publicly released in this repository at this stage.

| Artifact | Availability |
| --- | --- |
| Training metadata CSV | Not publicly released at this stage; available from the corresponding author upon reasonable request, subject to data availability restrictions |
| FASTA sequences | Not publicly released at this stage; available from the corresponding author upon reasonable request, subject to data availability restrictions |
| Generated ProtT5 LMDB embeddings | Not publicly released at this stage; can be regenerated with `python -m scripts.generate_embeddings` when sequence data are available |
| Trained checkpoint | Not publicly released at this stage; users can retrain the model with `python -m scripts.train` or provide their own checkpoint |
| Manuscript prediction outputs | Not publicly released at this stage; available from the corresponding author upon reasonable request |

Generated data, LMDB stores, checkpoints, logs, and experiment outputs are excluded from version control through `.gitignore`.

## Reproducibility Notes

The training script fixes the random seed with `seed_everything(42, workers=True)` by default. Full numerical reproducibility can still depend on the dataset split, ProtT5 model version, PyTorch and CUDA versions, GPU hardware, checkpoint selection, and preprocessing applied before LMDB generation.

The repository provides the code path needed to generate embeddings, train the regressor, and run inference. Because the manuscript data and trained checkpoint are not included, the public repository demonstrates the workflow and input formats but does not by itself reproduce the manuscript's numerical results.

## Citation

If you use this code, please cite:

Yu, C., Ji, M., He, J., Ma, H., and Liu, X. (2026). ProtACO: Prediction of Protein Absolute Contact Order from Sequence Using Pre-trained Language Models. Submitted manuscript.

The source code is available at `https://github.com/Diliraba777/ProtACO`. Publication details will be updated after acceptance if a DOI or formal citation becomes available.

## Acknowledgements

ProtACO was inspired by and partially adapted from TM-Vec, a method for protein remote homology detection and structural alignment. Please see `THIRD_PARTY_NOTICES.md` for the TM-Vec repository, paper, and BSD 3-Clause license notice.

## License

This project is released under the MIT License. Portions adapted from TM-Vec remain subject to the original TM-Vec BSD 3-Clause license notice; see `THIRD_PARTY_NOTICES.md` for details.
