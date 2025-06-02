import os, pickle
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
import torch
import tqdm
from pathlib import Path
import json
from transformers import (BitsAndBytesConfig, DefaultDataCollator)
from dataset import LogTokenizer, LogDataset, collate_fn
from models import SSModelForCLMWithLoRA
from typing import List, Callable
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Subset

seed = 42
torch.manual_seed(seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MetricUtils:
    @staticmethod
    def _compute_overall_perplexity(total_loss: float, total_tokens: int) -> float:
        if total_tokens == 0:
            return float('inf') # Or handle as an error, or return NaN
        average_loss_per_token = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(average_loss_per_token))
        return perplexity.item()

    @staticmethod
    def evaluate_perplexity(dataloader: DataLoader, model: SSModelForCLMWithLoRA, device: torch.device, output_dir: Path) -> dict: # Added device
        model.eval() # Set model to evaluation mode
        total_weighted_loss = 0.0  # Accumulate sum of losses (loss * num_tokens_in_batch)
        total_tokens_processed = 0 # Accumulate total number of actual (non-padded) tokens

        # To store per-example details if still desired for diagnostics
        per_example_details = {} 

        with torch.no_grad(): # IMPORTANT: Disable gradient calculations during evaluation
            with tqdm.tqdm(iterable=dataloader, desc="Evaluating Perplexity...", unit="example", colour="green") as pbar:
                for i, batch in enumerate(pbar): # Use enumerate for a simple index
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    
                    # No need to set model.eval() inside the loop, once at the beginning is enough
                    # No need for torch.autocast("cuda") if using torch.no_grad() unless mixed precision eval is explicitly desired
                    # For perplexity, full precision for loss calculation is usually fine.
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    
                    batch_loss = outputs["loss"] # This should be the average loss for the batch

                    # Count actual (non-ignored) tokens in the labels for this batch
                    # Assuming -100 is the ignore_index for padded labels
                    num_actual_tokens_in_batch = (labels != -100).sum().item()

                    if num_actual_tokens_in_batch > 0:
                        # Accumulate the sum of losses: batch_average_loss * number_of_tokens_in_batch
                        total_weighted_loss += batch_loss.item() * num_actual_tokens_in_batch
                        total_tokens_processed += num_actual_tokens_in_batch
                    
                    # Optional: Store per-example perplexity for diagnostics if batch_size is 1
                    if input_ids.size(0) == 1 and num_actual_tokens_in_batch > 0: # If batch size is 1
                        per_example_ppl = torch.exp(batch_loss).item() # PPL for this single example's average loss
                        source_text = model.tokenizer.decode(input_ids[0, :attention_mask[0].sum()].tolist()) # Decode only non-padded
                        per_example_details[i] = {
                            "id": i, 
                            "source_preview": source_text[:200] + "...", # Preview
                            "example_perplexity": f"{per_example_ppl:.2f}",
                            "example_avg_loss": f"{batch_loss.item():.4f}",
                            "num_tokens": num_actual_tokens_in_batch
                        }
                        
                    print(per_example_details[i])
                    if pbar.n % 10 == 0 or pbar.n == len(dataloader)-1:
                        # Calculate overall perplexity
                        overall_perplexity = MetricUtils._compute_overall_perplexity(total_weighted_loss, total_tokens_processed)
                        
                        print(f"\nOverall Test Perplexity: {overall_perplexity:.4f}")
                        print(f"Total Weighted Loss: {total_weighted_loss:.4f}")
                        print(f"Total Tokens Processed: {total_tokens_processed}")

                        results_summary = {
                            "overall_perplexity": f"{overall_perplexity:.4f}",
                            "total_weighted_loss": f"{total_weighted_loss:.4f}",
                            "total_tokens_processed": total_tokens_processed,
                            "per_example_diagnostics": per_example_details # Optional
                        }
                        
                        MetricUtils._save_results(str(output_dir), results_summary, "perplexity_summary")

        return results_summary
    
    @staticmethod
    def _save_results(output_dir: str, results: dict, metric_filename_prefix: str) -> None:
        # Ensure output_dir is a Path object for easier joining
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        filepath = output_dir_path / f"{metric_filename_prefix}_results.json" # Changed to .json for single dict
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {filepath}")

class LogEvaluator:
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

        self.log_tokenizer = self._get_tokenizer()
        self.Tmax = self._get_max_context_len()
        self.test_ds = LogDataset(
            log_tokenizer=self.log_tokenizer, 
            jsonl_dir=config["jsonl_dir"],
            cache_dir=config["cache_dir"],
            rebuild_index_cache=True,
            file_limit=max(1, int(config["frac"] * 11)),
            max_seq_len=self.Tmax
        )
        self.test_ds = Subset(dataset=self.test_ds, indices=range(int(len(self.test_ds) * config["frac"])))
        self.data_collator = lambda batch: collate_fn(batch, pad_token_id=self.log_tokenizer.tokenizer.pad_token_id, fixed_max_context_length=self.Tmax)
        self.dataloader = DataLoader(self.test_ds, batch_size=1, shuffle=False, collate_fn=self.data_collator)
    
        self.output_dir = Path(f"outputs/{self.config['task_name']}/{self.model_name_srt}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model = self._get_model()
    
    def _get_ckpt_config(self):
        config_pkl_path = Path.cwd() / self.config["checkpoint_path"] / "master_config.pkl"
        with open(config_pkl_path, 'rb') as f:
            config = pickle.load(f)
        return config
    
    def _get_tokenizer(self):
        config = self._get_ckpt_config()
        tokenizer = LogTokenizer(base_tokenizer_name=config["config"]["tokenizer_name"])
        return tokenizer
    
    def _get_max_context_len(self):
        config = self._get_ckpt_config()
        return config["config"]["max_context_len"]
    
    def _get_model(self):
        model_weights_file = Path.cwd() / self.config["checkpoint_path"] / self.config["checkpoint_name"] / "pytorch_model.bin"
        config = self._get_ckpt_config()

        model = SSModelForCLMWithLoRA(
            device=self.device,
            model_name=config["config"]["model_name"],
            tokenizer_name=config["config"]["tokenizer_name"],
            max_context_len=config["config"]["max_context_len"],
            grad_acc_steps=1,
            lora_alpha=config["config"]["lora_alpha"],
            lora_rank=config["config"]["lora_rank"],
            layer_names=config["config"]["target_modules"],
            quant_config=config["quant_config"],
        ).to(self.device)
          
        state_dict = torch.load(model_weights_file, map_location=self.device, weights_only=False) # weights_only=False for safety with custom classes
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"WARN: Missing keys when loading state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"INFO: Ignored unexpected keys when loading state_dict into model (likely quantization state)")
        print("Model initialized with weights.")
            
        return model
    
    def evaluate(self, metric: str) -> None:
        if metric == "perplexity":
            out_dict = MetricUtils.evaluate_perplexity(
                dataloader=self.dataloader,
                model=self.model,
                output_dir=self.output_dir,
                device=self.device
            )
            return out_dict
        else:
            raise ValueError(f"Unsupported metric: {metric}")

def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name, "task_name": "Log-Eval",
        "jsonl_dir": "data/dcn_jsonl_test", "cache_dir": "data/index_cache_test", 
        "checkpoint_path": "/mnt/home-ldap/mondal_ldap/proactiveRCA/outputs/ckpt/mamba-2.8b-hf_proactiveRCA/1.000_1.0e-05_r64_old", "checkpoint_name": "checkpoint-9160",
        "frac": 1.0, "is_4bit_quant": False,
    }
    evaluator = LogEvaluator(device=device, config=config)
    out = evaluator.evaluate(metric="perplexity") 
    print(out)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main("state-spaces/mamba-2.8b-hf", device=device) 