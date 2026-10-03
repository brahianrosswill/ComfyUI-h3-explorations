"""Per-head tau tables for the sparse attention node: the format, the loader, the refusals.

`MiniMaxH3Sol` takes one tau for every head of every block. Which heads
tolerate sparsity is a property of the model, not the scene: a head's routing
error keeps its rank from step to step and across prompt and seed
(`bench/results/2026-10-03_r2v_finish_time_budget.md`, "What a tau per head
would buy"), and it does not carry from one block to another. So the table is
one row per block, one value per head, calibrated once per checkpoint and
stage. The kernel takes it through `comfy_kitchen.sol_attn(tau_map=...)`
(`bench/results/2026-10-03_kitchen_tau_map.md`).

A table is one JSON file under `TABLE_DIR`:

    {
      "schema": 1,
      "provenance": {"calibrated_on": "...", "tool": "...", "date": "YYYY-MM-DD", ...},
      "heads": 56,
      "blocks": {"0": [56 taus], "24": [56 taus]}
    }

A block absent from `blocks` runs on the node's own `tau`. Any other key, a
row of the wrong length, a value that is not a finite number in
`[TAU_MIN, TAU_MAX]`, or a missing provenance field is refused, never
skipped: a table that half applies is a render nobody can describe.

Pure standard library, importable with no ComfyUI and no torch, so the checks
and the calibrator read tables the way the node does.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = 1
#: Where shipped tables live, one file per checkpoint and stage.
TABLE_DIR = Path(__file__).resolve().parent / "sparse_tables"
#: The node's "no table" choice.
NONE = "none"
#: A table's values are taus for the kernel's threshold, the same quantity as
#: the node's widget. **Reasoned**: 0 routes every block above the row mean;
#: the 2026-10-03 sweep's top value was 3.0 and almost nothing routes there,
#: so 8 is a ceiling that catches a typo without constraining a calibration.
TAU_MIN, TAU_MAX = 0.0, 8.0
#: What a table must say about where it came from.
PROVENANCE_REQUIRED = ("calibrated_on", "tool", "date")
_TOP_KEYS = {"schema", "provenance", "heads", "blocks"}


class SparseTableError(ValueError):
    """A table this pack will not apply."""


def list_tables(directory: Path | None = None) -> list[str]:
    """File names of the tables in `directory` (default `TABLE_DIR`), sorted."""
    d = TABLE_DIR if directory is None else Path(directory)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.json"))


def parse(doc, name: str = "<table>") -> dict:
    """Validate a decoded table. Returns `{"name", "heads", "blocks", "provenance",
    "sha256"}` with `blocks` as `{int: tuple of floats}` and `sha256` over the
    head count and the values."""
    if not isinstance(doc, dict):
        raise SparseTableError(f"{name}: a table is a JSON object, got {type(doc).__name__}")
    unknown = sorted(set(doc) - _TOP_KEYS)
    missing = sorted(_TOP_KEYS - set(doc))
    if unknown or missing:
        raise SparseTableError(f"{name}: unknown keys {unknown}, missing keys {missing}")
    if doc["schema"] != SCHEMA:
        raise SparseTableError(f"{name}: schema {doc['schema']!r}, this pack reads {SCHEMA}")
    prov = doc["provenance"]
    if not isinstance(prov, dict) or any(not str(prov.get(k, "")).strip() for k in PROVENANCE_REQUIRED):
        raise SparseTableError(f"{name}: provenance must name {list(PROVENANCE_REQUIRED)}")
    heads = doc["heads"]
    if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0:
        raise SparseTableError(f"{name}: heads must be a positive integer, got {heads!r}")
    raw = doc["blocks"]
    if not isinstance(raw, dict) or not raw:
        raise SparseTableError(f"{name}: blocks must be a non-empty object of block index to taus")
    blocks = {}
    for key, row in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise SparseTableError(f"{name}: block key {key!r} is not an integer") from None
        if index < 0 or str(index) != str(key):
            raise SparseTableError(f"{name}: block key {key!r} must be a plain non-negative integer")
        if not isinstance(row, list) or len(row) != heads:
            raise SparseTableError(f"{name}: block {index} needs {heads} taus, got "
                                   f"{len(row) if isinstance(row, list) else type(row).__name__}")
        values = []
        for h, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not TAU_MIN <= value <= TAU_MAX:
                raise SparseTableError(f"{name}: block {index} head {h} is {value!r}; a tau is a "
                                       f"finite number in [{TAU_MIN}, {TAU_MAX}]")
            values.append(float(value))
        blocks[index] = tuple(values)
    # The values' own hash, over a canonical form, so a record can say which
    # taus ran without carrying them: the file name alone does not.
    canon = json.dumps({"heads": heads, "blocks": {str(b): list(blocks[b]) for b in sorted(blocks)}},
                       sort_keys=True, separators=(",", ":"))
    return {"name": name, "heads": heads, "blocks": blocks, "provenance": dict(prov),
            "sha256": hashlib.sha256(canon.encode()).hexdigest()}


def load(name: str, directory: Path | None = None) -> dict:
    """Read and validate the table file `name` from `directory` (default `TABLE_DIR`)."""
    d = TABLE_DIR if directory is None else Path(directory)
    path = d / name
    if Path(name).name != name or not path.is_file():
        raise SparseTableError(f"no table {name!r} in {d.name}/ (it holds {list_tables(d)})")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SparseTableError(f"{name}: not JSON ({exc})") from None
    return parse(doc, name)


def file_fingerprint(name: str, directory: Path | None = None) -> str | None:
    """A hash of the table file's bytes, or None for `NONE` or a file that is
    not there. For a node's `fingerprint_inputs`: the calibrator rewrites a
    table under the same name, which changes no node input."""
    if not name or name == NONE or Path(name).name != name:
        return None
    try:
        return hashlib.sha256(((TABLE_DIR if directory is None else Path(directory)) / name).read_bytes()).hexdigest()
    except OSError:
        return None


def require_fits(table: dict, heads: int, n_blocks: int) -> None:
    """Refuse a table calibrated for another model: head count, or a block that does not exist."""
    if table["heads"] != heads:
        raise SparseTableError(f"{table['name']}: calibrated for {table['heads']} heads, "
                               f"this model has {heads}")
    beyond = sorted(b for b in table["blocks"] if b >= n_blocks)
    if beyond:
        raise SparseTableError(f"{table['name']}: names blocks {beyond}, this model has {n_blocks}")
