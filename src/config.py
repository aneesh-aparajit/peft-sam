from dataclasses import dataclass

@dataclass
class CFG:
    lora_rank: int = 32
    lora_alpha: int = 32
    target_modules = ["q_proj", "v_proj"]
    lora_dropout: float = 0.1
    bias: str = "lora_only"

    # Sam Params
    freeze_encoder: bool = True

    # training params
    batch_size = 32
    device = "cuda"
    num_epochs = 10
    gradient_accumulation = True
    gradient_accumulation_steps = 8
    learning_rate = 3e-4