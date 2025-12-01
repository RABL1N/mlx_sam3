import mlx.nn as nn
from sam3.model.necks import Sam3DualViTDetNeck

class SAM3VLBackbone(nn.Module):
    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp=0
    ):
        super().__init__()
        # TODO: check if we can compile like they do in PyTorch version
        self.vision_backbone: Sam3DualViTDetNeck = visual
        self.language_backbone = text
        self.scalp = scalp
        
        # TODO: Learn more about this from pytorch
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone
    
    def __call__(self):
        pass
        