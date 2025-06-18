import os
import wandb
if __name__ == "__main__":
    wandb.login()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pickle
import torch
from torch.utils.data import Subset
from transformers import (BitsAndBytesConfig, DefaultDataCollator,
                          TrainingArguments, Trainer,
                          get_linear_schedule_with_warmup)
from dataset import LogTokenizer, LogDataset, collate_fn
from models import SSModelForCLMWithLoRA

seed = 42
torch.manual_seed(seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LoRAFineTuner:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
        ) if self.config["is_4bit_quant"] else None
        
        self.model_name = config["model_name"]
        self.model_name_srt = self.model_name.split("/")[-1]
        self.log_tokenizer = LogTokenizer(base_tokenizer_name=config["tokenizer_name"])
        self.train_ds = LogDataset(log_tokenizer=self.log_tokenizer, 
                                   jsonl_dir=config["jsonl_dir"],
                                   cache_dir=config["cache_dir"],
                                   rebuild_index_cache=True,
                                   file_limit=max(1, int(config["frac"] * 70)),
                                   max_seq_len=config["max_context_len"])
        
        self.train_ds = Subset(dataset=self.train_ds, indices=range(int(len(self.train_ds) * config["frac"])))
        self.data_collator = lambda batch: collate_fn(batch, pad_token_id=self.log_tokenizer.tokenizer.pad_token_id, fixed_max_context_length=config["max_context_len"])

        self.batch_size = self.config["batch_size"]
        self.grad_acc_steps = self.config["grad_acc_steps"]
        self.num_steps = len(self.train_ds)//(self.batch_size * self.grad_acc_steps)
        self.num_epochs = self.config["num_epochs"]
        self.project_name = f"{self.model_name_srt}_{self.config['task_name']}"
        os.environ["WANDB_PROJECT"] = self.project_name
        self.run_name = f"{self.config['frac']:.3f}_{self.config['initial_lr']:.1e}_r{self.config['lora_rank']}"
        self.output_dir = f"{self.config['output_base_dir']}/outputs/ckpt/{self.project_name}/{self.run_name}"

        self.model = SSModelForCLMWithLoRA(
            device=self.device,
            model_name=self.model_name,
            tokenizer_name=self.config["tokenizer_name"],
            max_context_len=self.config["max_context_len"],
            grad_acc_steps=self.grad_acc_steps,
            lora_alpha=self.config["lora_alpha"],
            lora_rank=self.config["lora_rank"],
            layer_names=self.config["target_modules"],
            quant_config=self.quant_config,
        )
        
        # Unfreeze token embedding
        self.model.unfreeze_and_mask_embeddings(tokenizer_name=self.config["tokenizer_name"])
        
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                           lr=self.config['initial_lr'], 
                                           weight_decay=self.config["weight_decay"], 
                                           betas=self.config["adam_betas"])
        
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=int(0.01 * self.num_steps), 
                                                         num_training_steps=int(self.num_epochs * self.num_steps))

        self.train_arg_config = {
            "output_dir": self.output_dir,
            "eval_strategy": "no",
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.grad_acc_steps,
            "max_grad_norm": self.config["max_grad_norm"],
            "num_train_epochs": self.num_epochs,
            "logging_strategy": "steps",
            "logging_first_step": True,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": self.num_steps//self.config["num_ckpt_per_epoch"],
            "save_safetensors": False, 
            "save_total_limit": self.num_epochs * self.config["num_ckpt_per_epoch"],
            "save_only_model": False,
            "fp16": self.config["fp16"],
            "bf16": self.config["bf16"],
            "dataloader_drop_last": True,
            "run_name": self.run_name,
            "report_to": "wandb" if self.config["wandb_log"] else "none",
            "eval_on_start": False
        }
        
        self.training_args = TrainingArguments(**self.train_arg_config)
        self.trainer_config = {
            "model": self.model,
            "args": self.training_args,
            "data_collator": self.data_collator,
            "train_dataset": self.train_ds,
            "tokenizer": self.log_tokenizer.tokenizer,
            "optimizers": (self.optimizer, self.scheduler),
        }
        self.trainer = Trainer(**self.trainer_config)
    
    @staticmethod
    def compute_loss(self, model: torch.nn.Module, inputs: dict, outputs: torch.tensor = None):
        """Not to be used by LoRAFineTuner.
        Go to: ~/miniconda3/envs/env_name/lib/python3.8/site-packages/transformers/trainer.py
        Modify Trainer.compute_loss() function.
        Add this code snippet just before the return statement
        """
        # Custom: Begin.
        if model.training:
            total_norm = 0.0
            for p in model.parameters():
                if p.requires_grad:
                    param_norm = p.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5

            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"accuracy": outputs["acc"], "param_norm": total_norm})
        # Custom: End.
     
    def _save_config(self) -> None: 
        config_data = {
            "config": self.config,
            "output_dir": self.output_dir,
            "project_name": self.project_name,
            "run_name": self.run_name,
            "train_arg_config": self.train_arg_config,
            "quant_config": self.quant_config,
        }
        with open(f"{self.output_dir}/master_config.pkl", 'wb') as f:
            pickle.dump(config_data, f)
        print(f"Config saved to {self.output_dir}/master_config.pkl")

    def train(self) -> None:
        self._save_config()
        print(self.model)
        self.model.calc_num_params()
        self.model.train()
        self.trainer.train(resume_from_checkpoint=False)
        
def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name, "tokenizer_name": "EleutherAI/gpt-neox-20b", "task_name": "proactiveRCA", 
        "jsonl_dir": "data/dcn_jsonl_train", "cache_dir": "data/index_cache_train", "output_base_dir": "/mnt/commonfolder/networkAI/soumen/proactiveRCA_19",
        "num_epochs": 1, "batch_size": 1, "max_context_len": 6144, 
        "frac": 1.0, "target_modules": ['in_proj', 'x_proj', 'dt_proj', 'out_proj', 'lm_head'],
        "initial_lr": 1e-5, "lora_rank": 64, "lora_alpha": 128, "max_grad_norm": 10.0, "weight_decay": 0.1,
        "adam_betas": (0.95, 0.999), "grad_acc_steps": 16, "num_ckpt_per_epoch": 5, "is_4bit_quant": False, "fp16": False, "bf16": True,
        "wandb_log": True
    }
    trainer = LoRAFineTuner(device=device, config=config)
    trainer.train()
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main("state-spaces/mamba-2.8b-hf", device=device)