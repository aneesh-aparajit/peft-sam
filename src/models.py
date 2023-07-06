from typing import Dict, List

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import SamModel, SamProcessor

from config import CFG
from utils import get_bounding_boxes, print_trainable_params


# ---------------------------------------------------------------------------- #
#                              Dichotomous Dataset                             #
# ---------------------------------------------------------------------------- #
class Dis5kDataset(Dataset):
    def __init__(self, img_paths: List[str], mask_paths: List[str], processor: SamProcessor) -> None:
        super(Dis5kDataset, self).__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.processor = processor

        self.transforms = A.Compose([
            A.Resize(height=1024, width=1024),
            ToTensorV2()
        ])
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        img_pth = self.img_paths[ix]
        mask_pth = self.mask_paths[ix]

        img = np.array(Image.open(img_pth).convert('RGB'))
        mask = np.array(Image.open(mask_pth).convert('RGB'))

        transformed = self.transforms(image=img, mask=mask)

        img, mask = transformed['image'], transformed['mask']

        prompt = get_bounding_boxes(gt_map=mask)

        inputs = self.processor(images=img, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        inputs['gt'] = mask

        return inputs


# ---------------------------------------------------------------------------- #
#                                Build SamModel                                #
# ---------------------------------------------------------------------------- #
class LoraSamModel(nn.Module):
    '''This is the custom SAM model which will be used in the PEFT module'''
    def __init__(self, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.model = SamModel.from_pretrained("facebook/sam-vit-base")
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                    param.requires_grad = False
        self.config = self.model.config
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model.forward(
            pixel_values=batch['pixel_values'],
            input_boxes=batch['input_boxes'],
        )
        return outputs.pred_masks


def get_lora_sam_model(freeze: bool = False):
    '''Returns a Peft Model'''
    config = LoraConfig(
        r=CFG.lora_rank, 
        lora_alpha=CFG.lora_alpha, 
        target_modules=CFG.target_modules, 
        lora_dropout=CFG.lora_dropout, 
        bias=CFG.bias
    )
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    if freeze:
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad = False
    lora_model = get_peft_model(model, config)
    return lora_model



if __name__ == '__main__':
    print(f"* Normal SAM model (with all weights unfrozen):", end="\n  * ")
    model = LoraSamModel(freeze_encoder=False)
    print_trainable_params(model=model)

    print(f"* Normal SAM model (with encoder weights unfrozen):", end="\n  * ")
    model = LoraSamModel(freeze_encoder=True)
    print_trainable_params(model=model)

    print(f"* Normal peft SAM model (frozen encoders):", end="\n  * ")
    model = get_lora_sam_model(freeze=True)
    print_trainable_params(model=model)

    print(f"* Normal peft SAM model (un-frozen encoders):", end="\n  * ")
    model = get_lora_sam_model(freeze=False)
    print_trainable_params(model=model)
