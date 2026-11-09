"""Trainer for Otto's NNUE-style value network.

Modules:
  layout  - feature layout (read from the data files Rust writes) and the network architecture constants
  data    - memory-mapped `sim datagen` data sets, the train/validation split, a chunked, shuffled loader
  target  - the training target (outcome blended with a fitted teacher score)
  model   - the float model: factorised feature transformer + bucketed dense layers, quantisation-aware
  export  - quantisation, the `.nnue` file writer/reader, and an exact integer reference forward pass
  train   - the training CLI (`python -m otto_nnue.train`)
  parity  - checks Rust inference against the integer reference (`python -m otto_nnue.parity`)
"""
