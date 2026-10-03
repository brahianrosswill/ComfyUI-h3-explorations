# Sparse attention tables

One JSON file per checkpoint and stage: a tau for every head of the blocks it
lists, read by `MiniMaxH3Sol`'s `tau_table` input. `sparse_table.py` owns the
format and refuses anything it does not recognise; a block a table does not
list runs on the node's own `tau`.

No table ships yet. A table goes here only with its calibration record under
`bench/results/`, named in its `provenance`.
