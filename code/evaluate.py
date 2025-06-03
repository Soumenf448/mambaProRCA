import os, pickle
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
import torch
import tqdm
from pathlib import Path
import json
from dataset import LogTokenizer, LogDataset, collate_fn
from models import SSModelForCLMWithLoRA
from typing import List, Callable
from transformers import AutoTokenizer
from transformers import (BitsAndBytesConfig, DefaultDataCollator, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList) # Added StoppingCriteria

from typing import List, Callable, Dict, Any # Added Dict, Any
from torch.utils.data import Dataset, DataLoader, Subset

# For BLEU score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # pip install nltk

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
    def _generate_message_content(
        model: SSModelForCLMWithLoRA,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Generates message content until max_new_tokens or stop_token_id."""
        
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            "temperature": 0.7,
            "output_scores": True,
            "return_dict_in_generate": True,
            "eos_token_id": model.tokenizer.eos_token_id,
            "pad_token_id": model.tokenizer.pad_token_id,
            "stop_strings": [model.tokenizer.eos_token, "</msg>"]
        }
        out = model.generate(prompt=prompt, gen_config=gen_config)
        return out["output_text"]
    
    @staticmethod
    def _get_prompt_from_batch(batch: Dict[str, Any], tokenizer: AutoTokenizer) -> dict:
        """Extracts the prompt from the batch for generation."""
        assert batch["input_ids"].shape[0] == 1, "Batch size must be 1"
        input_ids = batch["input_ids"][0].tolist()
        start_msg_token_id = tokenizer.convert_tokens_to_ids("<msg>")
        end_msg_token_id = tokenizer.convert_tokens_to_ids("</msg>")
        
        # Find the start and end of the message
        start_idx = input_ids.index(start_msg_token_id) if start_msg_token_id in input_ids else 0
        end_idx = input_ids.index(end_msg_token_id) if end_msg_token_id in input_ids else len(input_ids)
        len_msg = end_idx - start_idx
        
        # Decode the input_ids to get the prompt
        prompt_tokens = input_ids[:start_idx+1]
        reference_tokens = input_ids[start_idx+1:end_idx]  # The message part
        prompt = tokenizer.decode(prompt_tokens)
        gt_message_str = tokenizer.decode(reference_tokens)
        return {"prompt": prompt, "len_msg": len_msg, "gt_message_str": gt_message_str}
        
    @staticmethod
    def evaluate_bleu(
        dataloader: DataLoader, 
        model: SSModelForCLMWithLoRA,
        device: torch.device,
        output_dir: Path,
        max_gt_message_tokens: int,
    ) -> Dict[str, Any]:
        
        model.eval()
        all_results_details = []
        bleu_scores = []
        processed_for_bleu_count = 0
        skipped_due_to_length_count = 0
        chencherry = SmoothingFunction() # For BLEU smoothing

        with torch.no_grad():
            with tqdm.tqdm(iterable=dataloader, desc="Evaluating BLEU Score...", unit="example", colour="cyan") as pbar:
                for i, batch in enumerate(pbar):
                    out = MetricUtils._get_prompt_from_batch(batch, model.tokenizer)
                    prompt = out["prompt"]
                    gt_message_str = out["gt_message_str"]
                    gt_message_token_len = out["len_msg"]
                     
                    if gt_message_token_len > max_gt_message_tokens:
                        skipped_due_to_length_count += 1
                        continue # Skip this example

                    processed_for_bleu_count += 1
                    generated_message_str = MetricUtils._generate_message_content(
                        model=model,
                        prompt=prompt,
                        max_new_tokens=gt_message_token_len + 20, # Allow some buffer for generation
                    )

                    reference_tokens = [gt_message_str.split()] # List of lists of tokens
                    hypothesis_tokens = generated_message_str.split()
                    
                    try:
                        bleu = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=chencherry.method1, weights=(0.25, 0.25, 0.25, 0.25))
                    except ZeroDivisionError: # Can happen if hypothesis is empty or too short
                        bleu = 0.0
                    except Exception as e_bleu:
                        print(f"Warning: BLEU calculation failed for example {i}: {e_bleu}")
                        bleu = 0.0 # Assign a score to avoid breaking aggregation
                        
                    bleu_scores.append(bleu)

                    all_results_details.append({
                        "id": i,
                        "gt_message": gt_message_str,
                        "generated_message": generated_message_str,
                        "bleu_score": bleu
                    })
                    
                    if i % 20 == 0: # Log progress occasionally
                        pbar.set_postfix_str(f"Avg BLEU so far: {sum(bleu_scores)/len(bleu_scores):.4f}")
                        average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
                        final_summary = {
                            "average_bleu_score": f"{average_bleu:.4f}",
                            "max_gt_message_tokens": max_gt_message_tokens,
                            "num_examples_evaluated_for_bleu": processed_for_bleu_count,
                            "num_examples_skipped_due_to_length": skipped_due_to_length_count,
                            "per_example_details": all_results_details # Optional: can make JSON large
                        }
                        MetricUtils._save_results(str(output_dir), final_summary, "bleu_score_summary")
        print(f"\nBLEU Score Evaluation Summary: {final_summary['average_bleu_score']}")
        return final_summary
    
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
        elif metric == "bleu":
            out_dict = MetricUtils.evaluate_bleu(
                dataloader=self.dataloader,
                model=self.model,
                output_dir=self.output_dir,
                device=self.device,
                max_gt_message_tokens=200,  # Adjust as needed
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
    out = evaluator.evaluate(metric="bleu") 
    print(out)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main("state-spaces/mamba-2.8b-hf", device=device) 