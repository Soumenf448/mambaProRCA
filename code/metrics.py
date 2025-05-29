import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import GenerationConfig
seed = 42
torch.manual_seed(seed)

class Metrics:
    
    @staticmethod
    def calc_num_params(named_parameters) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in named_parameters:
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"Training Percentage: {train_params * 100 / total_params:.3f}%")
        
    @staticmethod
    def compute_loss_ce(z: torch.Tensor, labels: torch.Tensor, grad_acc_steps: int) -> torch.Tensor:
        """
        Computes the CE loss.

        Args:
            z (torch.Tensor): The logits output from the model. Shape: (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): The ground truth labels. Shape: (batch_size, seq_len).
            grad_acc_steps (int): Gradient accumulation steps. Loss is divided by this factor.

        Returns:
            torch.Tensor: The computed loss value (scalar tensor).
        
        Raises:
            NotImplementedError: If the loss_name is not recognized.
        """
        if labels is None:
            raise ValueError("Labels cannot be None for loss computation.")
        # Standard cross-entropy loss for Causal Language Modeling.
        # z.view(-1, z.size(-1)) reshapes logits from (b, T, V) to (b*T, V).
        # labels.view(-1) reshapes labels from (b, T) to (b*T).
        # ignore_index=-100 ensures that tokens with label -100 (e.g., padding or prompt tokens)
        # do not contribute to the loss.
        loss = F.cross_entropy(z.view(-1, z.size(-1)), labels.view(-1), ignore_index=-100)
        
        # Scale loss by 1/grad_acc_steps for gradient accumulation.
        return (1 / grad_acc_steps) * loss

    @staticmethod
    def compute_acc(z: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Computes the accuracy, ignoring tokens where the label is -100 (e.g., prompt or padding).
        This method works for any batch size.

        Args:
            z (torch.Tensor): The logits output from the model. 
                              Shape: (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): The ground truth labels. 
                                   Shape: (batch_size, seq_len).
                                   Tokens to be ignored for accuracy (prompt, padding) 
                                   should be labeled as -100.
        Returns:
            float: The computed accuracy value (correct predictions / total valid tokens).
        
        Raises:
            ValueError: If labels tensor is None.
        """
        if labels is None:
            raise ValueError("Labels cannot be None for accuracy computation.")

        # Get model's predictions by taking argmax over the vocabulary dimension.
        predictions = z.argmax(dim=-1)  # Shape: (batch_size, seq_len)

        # Create a mask to identify valid tokens for accuracy calculation.
        # Valid tokens are those where the label is NOT -100.
        valid_token_mask = (labels != -100)  # Shape: (batch_size, seq_len), dtype=torch.bool

        # Select the predictions and labels only for these valid token positions.
        # Applying the boolean mask will flatten the selected elements into 1D tensors.
        valid_predictions = predictions[valid_token_mask]
        valid_labels = labels[valid_token_mask]
        num_valid_tokens = valid_labels.numel()
      
        # Compare the selected predictions with the selected labels and sum up the correct ones.
        correct_predictions_count = (valid_predictions == valid_labels).sum().item()
        
        # Accuracy is the number of correct predictions divided by the total number of valid tokens.
        accuracy = correct_predictions_count / num_valid_tokens
        
        return accuracy
    
    @staticmethod
    def _truncate_output(output_text: str, stop_strings: List[str]) -> str:
        earliest_index = None
        for stop in stop_strings:
            index = output_text.find(stop)
            if index != -1:
                if earliest_index is None or index < earliest_index:
                    earliest_index = index
        if earliest_index is not None:
            return output_text[:earliest_index]
        else:
            return output_text
    
    @staticmethod
    def generate(model, prompt: str, gen_config: dict) -> str:
        inputs = model.tokenizer(prompt, padding=False, truncation=True, max_length=model.Tmax, return_tensors="pt", add_special_tokens=True) # (Tmax,)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_config = GenerationConfig(**gen_config)
        model.model.eval()
        with torch.no_grad():
            output = model.model.generate(generation_config=gen_config, tokenizer=model.tokenizer, **inputs)
        output_text = model.tokenizer.decode(output.sequences.tolist()[0], skip_special_tokens=True)
        output_text_raw = output_text.replace(prompt, "").strip()
        output_text = Metrics._truncate_output(output_text_raw, stop_strings=gen_config.stop_strings)
        return {"prompt_text": prompt, "output_text": output_text, "output_text_raw": output_text_raw}
    