import mlx.nn as nn

from sam3.model.vl_combiner import SAM3VLBackbone

class Sam3Image(nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2
    
    def __init__(
        self,
        backbone: SAM3VLBackbone,
    ):
        super().__init__()
        self.backbone = backbone
    
    def __call__(self):
        pass