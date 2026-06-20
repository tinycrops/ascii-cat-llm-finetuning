# ASCII Cat Format Proofs

This folder contains two renderer proof passes for the cat ASCII training data.

## v1: local training data preservation

- `make_ascii_cat_training_animation.py`
- `ascii_cat_training_data_format_proof_45s.mp4`
- `format_proof_report.txt`
- `midpoint_still.png`

The first pass renders the local `src/dataset/ascii_art/animals/cat/*/content.txt`
files through the same `<ascii>...</ascii>` wrapper shape used by the training
script.

## v2: dimension coverage

- `make_ascii_cat_dimension_animation_v2.py`
- `ascii_cat_dimension_proof_v2_45s.mp4`
- `dimension_proof_v2_report.txt`
- `v2_wide_tall_still.png`

The second pass keeps the v1 script unchanged and adds adaptive rendering for
much wider and taller cat-family ASCII art. It combines:

- the local repo cat samples
- ASCIIArt.eu cat cards
- `apehex/ascii-art` cat-family rows from `asciiart/train/animals.parquet`
- a measured XMission dimension catalog with a dimension-stratified rendered
  subset

The v2 animation includes a width/height scatterplot to show the renderer's
coverage of the observed source dimensions.
