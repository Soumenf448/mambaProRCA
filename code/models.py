import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "9999"

import sys, pickle
from pathlib import Path

import torch
from dataset import LogTokenizer, LogDataset, collate_fn
from transformers import (AutoModelForCausalLM, MambaForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from typing import List, Dict, Tuple, Union
from metrics import Metrics
from torch.utils.data import DataLoader

seed = 42
torch.manual_seed(seed)

sys.path.append(Path(__file__).parent)

class SSModelForCLM(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, tokenizer_name: str, max_context_len: int, grad_acc_steps: int, quant_config: Union[BitsAndBytesConfig, None]):
        super(SSModelForCLM, self).__init__()
        self.device = device
        self.model_name = model_name
        self.quant_config = quant_config
        self.Tmax = max_context_len
        self.grad_acc_steps = grad_acc_steps
        self.tokenizer = LogTokenizer(base_tokenizer_name=tokenizer_name).tokenizer
        
        if "mamba" in self.model_name:
            self.model = MambaForCausalLM.from_pretrained(self.model_name, quantization_config=self.quant_config, device_map="auto")
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented.")
        
        # 3. Resize model embeddings if tokenizer vocab is larger
        current_vocab_size = self.model.get_input_embeddings().weight.size(0)
        if len(self.tokenizer) > current_vocab_size:
            print(f"Resizing token embeddings from {current_vocab_size} to {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 4. Set pad_token_id on the model's config and ensure tokenizer agrees
        if self.tokenizer.pad_token_id is None:
            print("Warning: Tokenizer pad_token_id is None. Attempting to use eos_token_id as fallback for model config if necessary.")
            if self.tokenizer.eos_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id # Common fallback
            else:
                print("Critical Warning: Tokenizer has no pad_token_id and no eos_token_id to fallback on for model config.")
        else:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.d = self.model.config.hidden_size
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        self.config = self.model.config
    
    def calc_num_params(self) -> None:
        Metrics.calc_num_params(self.named_parameters())
        
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: torch.tensor, **kwargs) -> torch.tensor:
        """all inputs shape = (b, T)"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device) if labels is not None else None
        
        with torch.autocast("cuda"):
            z = self.model(input_ids=input_ids, attention_mask=attention_mask).logits # (b, T, d)
        
        if labels is not None:
            loss = Metrics.compute_loss_ce(z, labels, grad_acc_steps=self.grad_acc_steps)
            acc = Metrics.compute_acc(z, labels)
        else:
            loss = None
            acc = None
       
        return {"logits": z, "loss": loss, "acc": acc}
    
    def generate(self, prompt: str, gen_config: dict):
        out = Metrics.generate(self, prompt=prompt, gen_config=gen_config)
        return out

class LoRALayer(torch.nn.Module):
    def __init__(self, rank: int, alpha: float, d_in: int, d_out: int):  
        super(LoRALayer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.alpha = alpha
        self.rank = rank
        
        self.A = torch.nn.Parameter(
            data=torch.normal(mean=0, std=0.01, size=(self.d_in, self.rank)), 
            requires_grad=True
        )
        self.B = torch.nn.Parameter(
            data=torch.zeros(size=(self.rank, self.d_out)),
            requires_grad=True
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        A = self.A.to(x.device)
        B = self.B.to(x.device)
        delta_W = torch.matmul(A, B) * (self.alpha / self.rank)
        z = torch.matmul(x, delta_W)
        return z
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        super(LinearWithLoRA, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = linear
        self.d_in = self.linear.in_features
        self.d_out = self.linear.out_features
        self.lora = LoRALayer(rank=self.rank, alpha=self.alpha, d_in=self.d_in, d_out=self.d_out)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        z1 = self.linear(x) 
        z2 = self.lora(x) 
        z = z1 + z2
        return z

    def __getattr__(self, name: str):
        """
        Forward attribute requests to the wrapped linear layer if not found on this module.
        This is crucial for compatibility with code that expects to access attributes
        like 'weight' or 'bias' from the original layer.
        """
        try:
            # Try to get attribute from this module (LinearWithLoRA) first
            return super().__getattr__(name)
        except AttributeError:
            # If not found on LinearWithLoRA, try to get it from the wrapped self.linear layer
            if hasattr(self.linear, name):
                return getattr(self.linear, name)
            else:
                # If also not found on self.linear, raise the AttributeError as it would normally.
                raise AttributeError(
                    f"'{type(self).__name__}' object and its wrapped '{type(self.linear).__name__}' "
                    f"object have no attribute '{name}'"
                )
    
class SSModelForCLMWithLoRA(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, tokenizer_name: str, max_context_len: int, grad_acc_steps: int, lora_rank: int, lora_alpha: float, layer_names: List[str], quant_config: Union[None, BitsAndBytesConfig]):
        super(SSModelForCLMWithLoRA, self).__init__()
        self.model_name = model_name
        self.device = device
        self.Tmax = max_context_len
        self.model = SSModelForCLM(device=self.device, model_name=self.model_name, tokenizer_name=tokenizer_name, max_context_len=self.Tmax, grad_acc_steps=grad_acc_steps, quant_config=quant_config)
        self.tokenizer = self.model.tokenizer
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.unfreeze_and_mask_embeddings(tokenizer_name=tokenizer_name) # Unfreeze and mask new token embeddings
        
        self.rank = lora_rank
        self.alpha = lora_alpha
        self.apply_lora(rank=self.rank, alpha=self.alpha, layer_names=layer_names)
        
    def unfreeze_and_mask_embeddings(self, tokenizer_name: str):
        """Unfreezes and applies gradient masks to train only new token embeddings."""
        try:
            # Load the base tokenizer to get its original size
            base_hf_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            original_vocab_size = len(base_hf_tokenizer)
            self._original_vocab_size_for_masking = original_vocab_size # Store for reference
            print(f"Original vocabulary size from '{tokenizer_name}': {original_vocab_size}")
        except Exception as e:
            print(f"ERROR: Could not load base tokenizer '{tokenizer_name}' to determine original_vocab_size: {e}")
            print("Cannot proceed with selective embedding training. Embeddings will remain as per initial freezing.")
            return

        current_full_vocab_size = len(self.tokenizer) # self.tokenizer is the augmented one
        num_new_tokens = current_full_vocab_size - original_vocab_size

        if num_new_tokens < 0:
            print(
                f"Warning: original_vocab_size ({original_vocab_size}) from '{tokenizer_name}' "
                f"is greater than the current augmented tokenizer's vocab size ({current_full_vocab_size}). "
                "This indicates a mismatch. Check tokenizer configurations. No embedding unfreezing will occur."
            )
            return
        
        if num_new_tokens == 0:
            print("No new tokens detected based on original_vocab_size. No selective embedding unfreezing needed.")
            return

        print(f"Preparing to selectively train embeddings for {num_new_tokens} new tokens.")

        # --- Handle Input Embeddings ---
        input_embeddings_layer = self.model.model.get_input_embeddings()
        if input_embeddings_layer is not None:
            print(f"Unfreezing and masking new input token embeddings (IDs {original_vocab_size} to {current_full_vocab_size-1}).")
            input_embeddings_layer.weight.requires_grad = True 

            new_token_ids_input = torch.arange(original_vocab_size, current_full_vocab_size, device=self.device)

            # Check if hook already exists to prevent double registration if called multiple times (though usually not)
            if not hasattr(input_embeddings_layer.weight, '_gradient_hook_input_embed'):
                def input_embedding_grad_hook(grad):
                    mask = torch.zeros_like(grad)
                    mask[new_token_ids_input] = 1.0 # new_token_ids_input is already on self.device
                    return grad * mask
                
                h = input_embeddings_layer.weight.register_hook(input_embedding_grad_hook)
                input_embeddings_layer.weight._gradient_hook_input_embed = h # Store handle to potentially remove later if needed
            else:
                print("Input embedding gradient hook already registered.")
        else:
            print("Warning: Could not get input_embeddings_layer for selective training.")
    
    def apply_lora(self, rank: int, alpha: float, layer_names: List[str]) -> None:
        SSModelForCLMWithLoRA.replace_linear_with_lora(device=self.device, model=self.model, rank=rank, alpha=alpha, layer_names=layer_names)            
     
    @staticmethod
    def replace_linear_with_lora(device: torch.device, model: torch.nn.Module, rank: int, alpha: float, layer_names: List[str]):
        """layer_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']"""
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                if any(proj in name for proj in layer_names):
                    linear_lora = LinearWithLoRA(module, rank, alpha)
                    setattr(model, name, linear_lora) # parent is model, child is module
            else:
                SSModelForCLMWithLoRA.replace_linear_with_lora(device, module, rank, alpha, layer_names)
     
    def calc_num_params(self) -> None:
        Metrics.calc_num_params(self.named_parameters())
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: Union[torch.tensor, None]) -> torch.tensor:
        assert list(input_ids.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
        return prediction_output

    def generate(self, prompt: str, gen_config: dict):
        out = Metrics.generate(self.model, prompt=prompt, gen_config=gen_config)
        return out

def main_clm(model_name: str, device: torch.device) -> None:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',  
        bnb_4bit_compute_dtype=torch.bfloat16,  
        bnb_4bit_use_double_quant=True,  
    ) if False else None
    Tmax = 1024
    # model = SSModelForCLM(device=device, model_name=model_name, tokenizer_name="EleutherAI/gpt-neox-20b", max_context_len=Tmax, grad_acc_steps=1, quant_config=quant_config).to(device)
    model = SSModelForCLMWithLoRA(device=device, model_name=model_name, tokenizer_name="EleutherAI/gpt-neox-20b", max_context_len=Tmax, grad_acc_steps=1, quant_config=quant_config, lora_rank=64, lora_alpha=128, layer_names=['in_proj', 'x_proj', 'dt_proj', 'out_proj', 'lm_head']).to(device)
    
    log_tokenizer = LogTokenizer(base_tokenizer_name="EleutherAI/gpt-neox-20b")
    log_dataset = LogDataset(
        log_tokenizer=log_tokenizer,
        jsonl_dir=Path(Path.cwd(), "data/dcn_jsonl"),
        cache_dir=Path(Path.cwd(), "data/log_dataset_item_cache"),
        rebuild_index_cache=False, # Set to True for the first run or if files change
        file_limit=None, # Process all files
        max_seq_len=Tmax
    )
    
    collate_function = lambda batch: collate_fn(batch, pad_token_id=log_tokenizer.tokenizer.pad_token_id, fixed_max_context_length=Tmax)
    dataloader = DataLoader(log_dataset, batch_size=4, shuffle=False, collate_fn=collate_function)
    
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device) # (b, T)
    labels = batch["labels"].to(device) # (b, T)
    attention_mask = batch["attention_mask"].to(device) # (b, T)
    
    print(model)
    model.calc_num_params()
    model.train()
    with torch.autocast("cuda"):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(out["logits"], out["loss"], out["acc"])
    
    print(model)
    model.eval()
    gen_config = {
        "max_new_tokens": 100,
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.7,
        "output_scores": True,
        "return_dict_in_generate": True,
        "eos_token_id": model.tokenizer.eos_token_id,
        "pad_token_id": model.tokenizer.pad_token_id,
        "stop_strings": [model.tokenizer.eos_token]
    }
    out = model.generate(prompt="Who is prime minister of India?", gen_config=gen_config)
    print(out)
    print("DONE")
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    main_clm("state-spaces/mamba-2.8b-hf", device=device) 
    
