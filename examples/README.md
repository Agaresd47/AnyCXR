# Example Cases

This directory contains small public-facing example cases derived from `Dataset003_Full/test_case`.

## Included Profiles

- `la`
- `pa`
- `oblique_45`

## Layout

- `inputs/`: sample chest radiographs
- `outputs/`: example segmentation outputs generated with the public CLI
- `example_cases.json`: metadata describing the bundled examples

## Regenerate

After downloading the HF checkpoint bundle to `./weights/anychest_inference_bundle.pt`, run:

```bash
bash scripts/build_example_cases.sh
```
