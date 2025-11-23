from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

@dataclass
class CHALLENGER_MaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    seg_logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None