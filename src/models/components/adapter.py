from typing import Optional, Tuple

import torch
import torch.nn as nn


class MemoryAdapter(nn.Module):
    def __init__(self, ltm: Optional[nn.Module] = None, stm: Optional[nn.Module] = None):
        super().__init__()

        self.ltm = ltm
        self.stm = stm
        example_ckpt_path = "data/resub/mm_30_20_512/checkpoints/last.ckpt"
        torch_ckpt = torch.load(example_ckpt_path, map_location="cpu")
