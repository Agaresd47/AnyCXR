# AnyCXR Public Inference

This repository is the inference-only public release for AnyCXR / AnyChest.

It is intended for reviewer and downstream use with:

- a local image file or folder of image files
- a local checkpoint bundle downloaded from Hugging Face
- no local training code
- no dependency on the original training workspace layout

## What is included

- `cxas/`: inference package
- `seg.py`: CLI entry point
- `cxas/data/anychest_reference.json`: packaged reference JSON with class order, view metadata, and the original `Dataset003_Full` folder map
- `pyproject.toml`, `setup.py`, `setup.cfg`: installation files

## Installation

Create a Python environment and install this folder:

```bash
pip install -e .
```

This installs the CLI command:

```bash
anychest-infer --help
```

## Download Checkpoints

Download the merged inference bundle from Hugging Face with the `hf` CLI:

```bash
hf download agaresd/anychest-inference anychest_inference_bundle.pt --local-dir ./weights
```

Model bundle repository:

- [agaresd/anychest-inference](https://huggingface.co/agaresd/anychest-inference)

The merged bundle contains three slimmed inference checkpoints:

- `la`
- `pa`
- `oblique`

Oblique profiles share one checkpoint and expose the following view-specific bundle profiles:

- `oblique_22_5`
- `oblique_45`
- `oblique_67_5`
- `oblique_112_5`
- `oblique_135`
- `oblique_157_5`

## Quick Start

Single LA radiograph:

```bash
anychest-infer \
  --input-path /path/to/image.png \
  --output-dir /path/to/output \
  --checkpoint ./weights/anychest_inference_bundle.pt \
  --profile la
```

Single PA DICOM:

```bash
anychest-infer \
  --input-path /path/to/image.dcm \
  --output-dir /path/to/output \
  --checkpoint ./weights/anychest_inference_bundle.pt \
  --profile pa
```

Single oblique image:

```bash
anychest-infer \
  --input-path /path/to/image.jpg \
  --output-dir /path/to/output \
  --checkpoint ./weights/anychest_inference_bundle.pt \
  --profile oblique_45
```

Flat folder of PA images:

```bash
anychest-infer \
  --input-path /path/to/folder \
  --output-dir /path/to/output \
  --checkpoint ./weights/anychest_inference_bundle.pt \
  --profile pa
```

Dataset-style folder tree:

```bash
anychest-infer \
  --input-path /path/to/Dataset003_Full \
  --output-dir /path/to/output \
  --checkpoint ./weights/anychest_inference_bundle.pt \
  --profile oblique
```

## Inputs

The public CLI accepts:

- `.png`
- `.jpg`
- `.jpeg`
- `.dcm`
- `.dicom`

`--input-path` can point to:

- a single file
- a flat folder of supported files
- a dataset-style folder containing view subfolders such as `LA/imagesTr` or `45/imagesTr`

If the input is flat or ambiguous, pass `--view-name`.

## Outputs

For each processed image the package writes:

- per-class segmentation masks under `labelsTr/`
- a color overlay under `overlays/`

Mask layout is controlled by:

- `--save-option one|sep|total`
- `--save-format img|npy`

## Reference JSON

The packaged reference JSON is [`cxas/data/anychest_reference.json`](cxas/data/anychest_reference.json).

It records:

- the 54 output classes in inference order
- the AnyChest view-to-angle mapping
- the original `Dataset003_Full` folder names such as `225`, `45`, `LA`, and `PA`

## Notes

- This release is inference-only. Training scripts and trainer code are intentionally excluded.
- The bundled reference JSON replaces the need for a local `dataset.json` during public inference.
- The merged HF bundle is slimmed to inference weights only; optimizer and scheduler state are removed.
