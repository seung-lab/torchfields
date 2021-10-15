import torch
from .fields import DisplacementField as Field
from .fields import set_identity_mapping_cache

torch.Field = Field
