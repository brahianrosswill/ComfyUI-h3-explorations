# Copied from facebookresearch/sam3 at commit 2345a4ad109ac29c569da749c91d84f10dc08c40 into ComfyUI-h3-explorations.
# Under the SAM License (meta_sam3/LICENSE), not this pack's licence. Differences from upstream:
# this header, relative imports, and the edits recorded in meta_sam3/EDITS.diff.
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os

is_enabled = False
if os.getenv("USE_PERFLIB", "1") == "1":
    # print("Enabled the use of perflib.\n", end="")
    is_enabled = True
