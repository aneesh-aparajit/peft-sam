import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from config import CFG
from metrics import Metrics
from models import get_lora_sam_model
from utils import print_trainable_params


class Trainer:
    def __init__(self, train_dataloader, valid_dataloader) -> None:
        self.model = get_lora_sam_model(freeze=CFG.freeze_encoder).to(CFG.device)
        print_trainable_params(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.learning_rate)
        self.metrics = Metrics()
        self.run = wandb.init(dir="../logs/", config=CFG.__dict__, project="sam-dis5k", tags=['segmentation', 'sam', 'dichotomous'])
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
    
    def train_step(self, batch, batch_ix):
        outputs = self.model.forward(pixel_values=batch['pixel_values'], input_boxes=batch['input_boxes'], multimask_output=False)
        preds = outputs.pred_masks.squeeze(1)
        metrics = self.metrics.get_metrics(pred=preds, gt=batch['gt'])
        return metrics
    
    @torch.no_grad()
    def valid_step(self, batch, batch_ix):
        outputs = self.model.forward(pixel_values=batch['pixel_values'], input_boxes=batch['input_boxes'], multimask_output=False)
        preds = outputs.pred_masks.squeeze(1)
        loss = F.binary_cross_entropy(preds, batch['gt'])
        metrics = self.metrics.get_metrics(pred=preds, gt=batch['gt'])
        metrics['valid_loss'] = loss
        return metrics

    def fit(self):
        best_dice = 0
        for epoch in range(CFG.num_epochs):
            print('#'*15)
            print(f'### Epoch {epoch+1}/{CFG.num_epochs}')
            print(f'#'*15)

            # ------------------------------- Training Loop ------------------------------ #
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc='(train)')
            self.model.train()
            for step, batch in pbar:
                batch = {k:v.to(CFG.device) for k, v in batch.items()}

                metrics = self.train_step(batch=batch, batch_ix=step)

                loss = metrics['train_loss']
                loss.backward()

                if CFG.gradient_accumulation:
                    if step % CFG.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                pbar.set_postfix({k:f'{v:.3f}' for k, v in metrics.items() if "train" not in k})
            
            # ------------------------------ Validation Loop ----------------------------- #
            self.model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader), desc='(valid)')
                for step, batch in pbar:
                    batch = {k:v.to(CFG.device) for k, v in batch.items()}
                    metrics = self.valid_step(batch=batch, batch_ix=step)
                    pbar.set_postfix({k:f'{v:.3f}' for k, v in metrics.items()})
            
            if metrics['hard_dice'] > best_dice:
                print(f"New best dice score achieved from {best_dice:.4f} to {metrics['hard_dice']:.4f}")
                best_dice = metrics['hard_dice']
                file_name = f"../artifacts/valid_bce={metrics['bce']}-valid_hard_dice={metrics['hard_dice']}=valid_soft_dice={metrics['soft_dice']}-valid_hard_iou={metrics['hard_iou']}=valid_soft_iou={metrics['soft_iou']}.pth"
                self.model.save_pretrained(save_directory="../artifacts/")
                print(f"Checkpoint saved at ../artifacts/")
        
        return self.model.state_dict()
