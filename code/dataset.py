import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import shutil # For creating/removing a temporary directory for cache if needed for testing
import pickle
import os
import collections # For Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

try:
    import matplotlib.pyplot as plt
    import numpy as np # Often used with matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger = logging.getLogger(__name__) # Ensure logger is defined if matplotlib fails
    logger.warning("matplotlib or numpy not found. Histogram will be text-based.")

# --- Standard Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)


class LogTokenizer:
    def __init__(self, base_tokenizer_name: str = "EleutherAI/gpt-neox-20b"):
        logger.info(f"Loading base tokenizer from: {base_tokenizer_name}")
        try:
            # Load the base tokenizer
            self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
                base_tokenizer_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Error loading base tokenizer '{base_tokenizer_name}': {e}")
            raise

        # Define the user-specified special tokens and their corresponding JSON keys
        self.field_to_tags_map = {
            "timestamp_utc": ("<time>", "</time>"),
            "log_type_hint": ("<ltype>", "</ltype>"),
            "message_content": ("<msg>", "</msg>"),
            "warrior_context_tag": ("<ctx>", "</ctx>"),
            "keyword_name": ("<kwrd>", "</kwrd>"),
            "step_id": ("<sid>", "</sid>"),
            "step_description": ("<sdesc>", "</sdesc>"),
            "step_status": ("<ssts>", "</ssts>"),
            "test_case_id": ("<tcid>", "</tcid>"),
            "test_case_status": ("<tcsts>", "</tcsts>"),
            "test_suite_id": ("<tsid>", "</tsid>"),
            "test_suite_status": ("<tssts>", "</tssts>"),
        }

        # Global entry delimiters
        self.entry_start_token = "<start>"
        self.entry_end_token = "</end>"

        # --- Corrected Special Token Handling ---
        # Start with the custom XML-like tags
        tokens_to_consider_adding_as_special = {self.entry_start_token, self.entry_end_token}
        for start_tag, end_tag in self.field_to_tags_map.values():
            tokens_to_consider_adding_as_special.add(start_tag)
            tokens_to_consider_adding_as_special.add(end_tag)

        # Define standard special tokens that we want to ensure exist
        standard_special_tokens_to_ensure = {}
        if self.tokenizer.eos_token is None:
            standard_special_tokens_to_ensure['eos_token'] = "[EOS]"
        if self.tokenizer.bos_token is None:
            standard_special_tokens_to_ensure['bos_token'] = "[BOS]"
        # Pad token logic: use EOS if available and no PAD, else use a new PAD token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                standard_special_tokens_to_ensure['pad_token'] = self.tokenizer.eos_token
            elif 'eos_token' in standard_special_tokens_to_ensure: # if EOS was just added
                 standard_special_tokens_to_ensure['pad_token'] = standard_special_tokens_to_ensure['eos_token']
            else:
                standard_special_tokens_to_ensure['pad_token'] = "[PAD]"
        
        # Add these standard tokens to our set if they are new definitions
        for token_val in standard_special_tokens_to_ensure.values():
            tokens_to_consider_adding_as_special.add(token_val)

        # Now, determine which of these are truly new to the tokenizer
        current_vocab_and_all_special_tokens = set(self.tokenizer.get_vocab().keys())
        current_vocab_and_all_special_tokens.update(self.tokenizer.all_special_tokens)
        
        newly_added_tokens = sorted([
            token for token in list(tokens_to_consider_adding_as_special)
            if token not in current_vocab_and_all_special_tokens
        ])

        if newly_added_tokens:
            logger.info(f"Adding {len(newly_added_tokens)} new special tokens overall: {newly_added_tokens}")
            # We need to pass the dictionary format if we are setting roles like eos_token
            # For additional_special_tokens, a list is fine.
            # It's safer to add them grouped if they are standard roles, then add others.
            
            # Add standard roles first if they were identified as new
            standard_tokens_actually_added_dict = {}
            if 'eos_token' in standard_special_tokens_to_ensure and standard_special_tokens_to_ensure['eos_token'] in newly_added_tokens:
                standard_tokens_actually_added_dict['eos_token'] = standard_special_tokens_to_ensure['eos_token']
            if 'bos_token' in standard_special_tokens_to_ensure and standard_special_tokens_to_ensure['bos_token'] in newly_added_tokens:
                standard_tokens_actually_added_dict['bos_token'] = standard_special_tokens_to_ensure['bos_token']
            if 'pad_token' in standard_special_tokens_to_ensure and standard_special_tokens_to_ensure['pad_token'] in newly_added_tokens:
                 # Handle case where pad_token is set to eos_token which might already exist or be newly added
                if standard_special_tokens_to_ensure['pad_token'] != self.tokenizer.eos_token and \
                   standard_special_tokens_to_ensure['pad_token'] != (standard_special_tokens_to_ensure.get('eos_token')):
                    standard_tokens_actually_added_dict['pad_token'] = standard_special_tokens_to_ensure['pad_token']
                elif self.tokenizer.pad_token is None: # if pad_token is still None after potential eos_token assignment
                     self.tokenizer.pad_token = self.tokenizer.eos_token # set it explicitly
                     logger.info(f"Set pad_token to existing/new eos_token: {self.tokenizer.eos_token}")


            if standard_tokens_actually_added_dict:
                self.tokenizer.add_special_tokens(standard_tokens_actually_added_dict)
                logger.info(f"Added standard missing tokens: {standard_tokens_actually_added_dict}")

            # Add the remaining XML-like tags
            other_custom_tags = [
                token for token in newly_added_tokens 
                if token not in standard_tokens_actually_added_dict.values()
            ]
            if other_custom_tags:
                self.tokenizer.add_special_tokens({"additional_special_tokens": other_custom_tags})
                logger.info(f"Added custom XML-like tags: {other_custom_tags}")

            logger.info(f"Tokenizer vocabulary size after all additions: {len(self.tokenizer)}")
        else:
            logger.info("No new unique special tokens needed to be added to the tokenizer vocabulary.")

        # Explicitly set tokenizer properties if they were added or defaulted
        if self.tokenizer.eos_token is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.eos_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
        if self.tokenizer.bos_token is None and self.tokenizer.bos_token_id is not None:
            self.tokenizer.bos_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.bos_token_id)
        if self.tokenizer.pad_token is None and self.tokenizer.pad_token_id is not None: # If pad_token was added as a new token
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
        elif self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None: # Default to EOS if still None
            logger.info(f"Setting pad_token to eos_token ('{self.tokenizer.eos_token}') as a final fallback.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None: # If tokenizer didn't auto-set pad_token_id
                 self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


        logger.info(f"Final tokenizer config -> BOS: '{self.tokenizer.bos_token}' (ID: {self.tokenizer.bos_token_id}), EOS: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id}), PAD: '{self.tokenizer.pad_token}' (ID: {self.tokenizer.pad_token_id})")
        logger.info(f"Final vocabulary size: {len(self.tokenizer)}")
    
    def serialize_log_entry(self, log_entry: Dict[str, Any]) -> str:
        serialized_parts = [self.entry_start_token]
        for field_key, (start_tag, end_tag) in self.field_to_tags_map.items():
            if field_key in log_entry and log_entry[field_key] is not None:
                value = str(log_entry[field_key])
                if field_key == "message_content":
                    value = value.replace("<NL>", "\n") 
                value = value.replace("<", "<").replace(">", ">")
                serialized_parts.append(f" {start_tag}{value}{end_tag}")
        
        serialized_parts.append(f" {self.entry_end_token}")
        return "".join(serialized_parts).strip() + self.tokenizer.eos_token

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)

    def batch_encode_plus(self, texts: List[str], **kwargs) -> Dict[str, List[List[int]]]:
        return self.tokenizer(texts, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)


class LogDataset(Dataset):
    def __init__(self,
                 log_tokenizer: LogTokenizer,
                 jsonl_dir: Path,
                 cache_dir: Path,
                 rebuild_index_cache: bool = False,
                 file_limit: Optional[int] = None,
                 max_seq_len: Optional[int] = None # For truncating individual lines if too long
                ):
        self.log_tokenizer_wrapper = log_tokenizer
        self.tokenizer = log_tokenizer.tokenizer # The actual Hugging Face tokenizer
        self.jsonl_dir = jsonl_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.file_limit = file_limit
        self.max_seq_len = max_seq_len

        # Index file maps a global index to (file_path, line_offset_in_bytes)
        self.index_cache_file = self.cache_dir / f"log_line_index_flimit{file_limit or 'all'}.pkl"
        self.line_index: List[Dict[str, Any]] = [] # List of {'file': Path, 'offset': int, 'length': int}

        if not rebuild_index_cache and self.index_cache_file.exists():
            logger.info(f"Loading line index from cache: {self.index_cache_file}")
            self._load_index_from_cache()
        else:
            logger.info(f"Index cache not found or rebuild_cache=True. Building index for: {self.jsonl_dir}")
            self._build_and_cache_index()

        if not self.line_index:
             logger.warning("Line index is empty. The dataset will also be empty.")

    def _load_index_from_cache(self):
        try:
            with open(self.index_cache_file, "rb") as f:
                self.line_index = pickle.load(f)
            logger.info(f"Successfully loaded {len(self.line_index)} line references from index cache.")
        except Exception as e:
            logger.error(f"Error loading index from cache: {e}. Rebuilding index.")
            self._build_and_cache_index()

    def _build_and_cache_index(self):
        self.line_index = []
        jsonl_files = sorted(list(self.jsonl_dir.glob("*.jsonl")))

        if not jsonl_files:
            logger.warning(f"No .jsonl files found in {self.jsonl_dir} to build index.")
            return

        if self.file_limit is not None:
            jsonl_files = jsonl_files[:self.file_limit]
            logger.info(f"Limiting index building to first {len(jsonl_files)} files.")
        
        current_global_line_idx = 0
        for file_path in tqdm(jsonl_files, desc="Building line index"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Instead of byte offset, we'll store file and line number
                    # This is simpler for on-demand reading but might be slower for very large files
                    # if we have to iterate to the line. A true byte offset index is more performant.
                    # For now, let's stick to (file_path, line_number_in_file) for simplicity of implementation.
                    for line_num_in_file, line_content in enumerate(f):
                        if line_content.strip(): # Only index non-empty lines
                            self.line_index.append({
                                'file_path': file_path, 
                                'line_number_in_file': line_num_in_file # 0-based
                            })
                            current_global_line_idx += 1
            except Exception as e:
                logger.error(f"Error reading or indexing file {file_path}: {e}")
        
        logger.info(f"Built index with {len(self.line_index)} total processable log lines.")
        try:
            with open(self.index_cache_file, "wb") as f:
                pickle.dump(self.line_index, f)
            logger.info(f"Line index cached to: {self.index_cache_file}")
        except Exception as e:
            logger.error(f"Error saving line index to cache: {e}")

    def __len__(self):
        return len(self.line_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self.line_index):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.line_index)}")

        index_entry = self.line_index[idx]
        file_path = index_entry['file_path']
        target_line_num = index_entry['line_number_in_file']
        
        line_content_str = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, current_line in enumerate(f):
                    if i == target_line_num:
                        line_content_str = current_line.strip()
                        break
        except Exception as e:
            logger.error(f"Error reading line {target_line_num} from file {file_path}: {e}")
            # Return empty tensors or raise error; for now, let's return placeholder for robustness in dataloader
            # This should be handled by the collate_fn if it expects consistent dict keys
            return {"input_ids": torch.tensor([], dtype=torch.long), "labels": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}


        if not line_content_str:
            logger.warning(f"Empty line content retrieved for index {idx} (file: {file_path}, line: {target_line_num})")
            return {"input_ids": torch.tensor([], dtype=torch.long), "labels": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}

        try:
            log_entry_dict = json.loads(line_content_str)
        except json.JSONDecodeError:
            logger.warning(f"Malformed JSON at index {idx} (file: {file_path}, line: {target_line_num}): {line_content_str[:100]}...")
            return {"input_ids": torch.tensor([], dtype=torch.long), "labels": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}

        serialized_log_str = self.log_tokenizer_wrapper.serialize_log_entry(log_entry_dict)
        
        # For next token prediction, input_ids and labels are shifted
        # Tokenize with BOS and EOS for each individual sequence
        full_token_ids = self.tokenizer.encode(
            serialized_log_str, 
            add_special_tokens=True, # Let tokenizer add BOS/EOS
            truncation=True if self.max_seq_len else False,
            max_length=self.max_seq_len if self.max_seq_len else None
        )

        if len(full_token_ids) < 2 : # Need at least BOS + one token to make a pair
            # This can happen if serialized_log_str is empty or only contains tokens that get filtered out by tokenizer
            logger.warning(f"Tokenized sequence too short for index {idx} (file: {file_path}, line: {target_line_num}). Serialized: '{serialized_log_str[:100]}...', Tokens: {full_token_ids}")
            return {"input_ids": torch.tensor([], dtype=torch.long), "labels": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}


        input_ids = full_token_ids[:-1]
        labels = full_token_ids[1:]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate function to pad sequences in a batch to the max length in that batch.
    """
    input_ids_list = [item['input_ids'] for item in batch if item['input_ids'].numel() > 0] # Filter empty tensors
    labels_list = [item['labels'] for item in batch if item['labels'].numel() > 0]

    if not input_ids_list: # If all items in batch were empty
        return {
            "input_ids": torch.empty(0,0, dtype=torch.long), 
            "labels": torch.empty(0,0, dtype=torch.long),
            "attention_mask": torch.empty(0,0, dtype=torch.long)
        }

    # Pad input_ids
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    
    # Create attention mask for input_ids
    attention_mask = (input_ids_padded != pad_token_id).long()
    
    # Pad labels. For causal LM, labels are usually padded with -100 (ignored by loss function)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100) 
    
    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask
    }

def find_max_context_length_for_logs(jsonl_dir: Path, log_tokenizer: LogTokenizer, file_limit: Optional[int] = None):
    """
    Finds the JSONL line with the maximum number of characters across all specified files,
    serializes it, tokenizes it, and prints information about its length.

    Args:
        jsonl_dir (Path): Directory containing the .jsonl log files.
        log_tokenizer (LogTokenizer): Initialized instance of LogTokenizer.
        file_limit (Optional[int]): Limits the number of JSONL files processed.
    """
    if not jsonl_dir.is_dir():
        logger.error(f"Provided JSONL directory does not exist: {jsonl_dir}")
        return

    jsonl_files = sorted(list(jsonl_dir.glob("*.jsonl")))
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {jsonl_dir}")
        return

    if file_limit is not None:
        jsonl_files = jsonl_files[:file_limit]
        logger.info(f"Analyzing up to {len(jsonl_files)} files.")

    max_char_len_raw = 0
    longest_raw_line_content = ""
    longest_raw_line_file = None
    longest_raw_line_num = -1

    logger.info("Scanning files to find the longest raw JSONL line by character count...")
    for file_path in tqdm(jsonl_files, desc="Scanning files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line_str in enumerate(f):
                    line_char_len = len(line_str)
                    if line_char_len > max_char_len_raw:
                        max_char_len_raw = line_char_len
                        longest_raw_line_content = line_str.strip()
                        longest_raw_line_file = file_path
                        longest_raw_line_num = i + 1
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            continue

    if not longest_raw_line_content:
        logger.warning("No processable lines found in the JSONL files.")
        return

    logger.info(f"\n--- Longest Raw JSONL Line Found ---")
    logger.info(f"File: {longest_raw_line_file}")
    logger.info(f"Line Number: {longest_raw_line_num}")
    logger.info(f"Character Length (raw line): {max_char_len_raw}")
    # logger.info(f"Raw Content (first 200 chars): {longest_raw_line_content[:200]}...")

    try:
        log_entry_dict = json.loads(longest_raw_line_content)
    except json.JSONDecodeError as e:
        logger.error(f"Could not parse the longest line as JSON: {e}")
        logger.error(f"Content was: {longest_raw_line_content[:500]}...")
        return

    logger.info("\n--- Serializing and Tokenizing Longest Line ---")
    serialized_string = log_tokenizer.serialize_log_entry(log_entry_dict)
    char_len_serialized = len(serialized_string)

    # Tokenize: For estimating max context, we should include BOS/EOS if the model
    # processes each entry as a full sequence framed by them.
    token_ids = log_tokenizer.encode(serialized_string, add_special_tokens=True) # Add BOS/EOS
    num_tokens = len(token_ids)

    logger.info(f"Character Length (serialized): {char_len_serialized}")
    logger.info(f"Number of Tokens (for this single longest entry, including BOS/EOS): {num_tokens}")
    logger.info(f"This token count ({num_tokens}) is an approximate upper bound for context length "
                f"if each log entry is treated as an independent sequence.")
    logger.info("For pretraining where sequences are concatenated and chunked, the chosen context length "
                "can be independent of any single line's tokenized length, but this gives "
                "an idea of the complexity of individual entries.")

def analyze_and_plot_token_length_distribution(
    jsonl_dir: Path, 
    log_tokenizer: LogTokenizer, 
    file_limit: Optional[int] = None,
    max_token_len_for_hist: int = 4096 # Cap histogram display range
    ):
    """
    Analyzes all JSONL files in a directory to get the distribution of
    tokenized log entry lengths and plots a histogram.

    Args:
        jsonl_dir (Path): Directory containing the .jsonl log files.
        log_tokenizer (LogTokenizer): Initialized instance of LogTokenizer.
        file_limit (Optional[int]): Limits the number of JSONL files processed.
        max_token_len_for_hist (int): Max token length to consider for histogram binning.
    """
    if not jsonl_dir.is_dir():
        logger.error(f"Provided JSONL directory does not exist: {jsonl_dir}")
        return

    jsonl_files = sorted(list(jsonl_dir.glob("*.jsonl")))
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {jsonl_dir}")
        return

    if file_limit is not None:
        jsonl_files = jsonl_files[:file_limit]
        logger.info(f"Analyzing up to {len(jsonl_files)} files for token length distribution.")

    all_token_lengths: List[int] = []

    logger.info("Processing all JSONL lines to get token lengths...")
    for file_path in tqdm(jsonl_files, desc="Analyzing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_str in f:
                    if not line_str.strip():
                        continue
                    try:
                        log_entry_dict = json.loads(line_str.strip())
                        serialized_string = log_tokenizer.serialize_log_entry(log_entry_dict)
                        # Tokenize with BOS/EOS as each entry is considered a sequence here
                        token_ids = log_tokenizer.encode(serialized_string, add_special_tokens=True)
                        all_token_lengths.append(len(token_ids))
                    except json.JSONDecodeError:
                        logger.debug(f"Skipping malformed JSON line in {file_path}")
                    except Exception as e:
                        logger.debug(f"Error tokenizing line in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            continue
    
    if not all_token_lengths:
        logger.warning("No token lengths were recorded. Cannot generate distribution.")
        return

    logger.info(f"\n--- Token Length Distribution Analysis (for {len(all_token_lengths)} log entries) ---")
    
    # Summary Statistics
    if MATPLOTLIB_AVAILABLE: # numpy is imported if matplotlib is
        lengths_np = np.array(all_token_lengths)
        logger.info(f"Min token length: {np.min(lengths_np)}")
        logger.info(f"Max token length: {np.max(lengths_np)}")
        logger.info(f"Mean token length: {np.mean(lengths_np):.2f}")
        logger.info(f"Median token length: {np.median(lengths_np)}")
        logger.info(f"90th percentile: {np.percentile(lengths_np, 90):.2f}")
        logger.info(f"95th percentile: {np.percentile(lengths_np, 95):.2f}")
        logger.info(f"99th percentile: {np.percentile(lengths_np, 99):.2f}")
        logger.info(f"99.9th percentile: {np.percentile(lengths_np, 99.9):.2f}")
    else: # Basic stats without numpy
        all_token_lengths.sort()
        logger.info(f"Min token length: {all_token_lengths[0] if all_token_lengths else 'N/A'}")
        logger.info(f"Max token length: {all_token_lengths[-1] if all_token_lengths else 'N/A'}")
        mean_len = sum(all_token_lengths) / len(all_token_lengths) if all_token_lengths else 0
        logger.info(f"Mean token length: {mean_len:.2f}")
        # (Percentiles are harder without numpy/scipy, so skipping for text-only)

    # Plotting with Matplotlib
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 7))
        # Filter lengths for histogram display range to avoid extremely long tails making plot unreadable
        display_lengths = [l for l in all_token_lengths if l <= max_token_len_for_hist]
        if not display_lengths: # If all lengths are > max_token_len_for_hist
            display_lengths = all_token_lengths # Show all in this case

        plt.hist(display_lengths, bins=50, edgecolor='black', alpha=0.7) # Adjust bins as needed
        plt.title(f'Histogram of Tokenized Log Entry Lengths (Capped at {max_token_len_for_hist} for display)')
        plt.xlabel('Number of Tokens per Log Entry (including BOS/EOS)')
        plt.ylabel('Number of Log Entries')
        plt.grid(axis='y', alpha=0.75)
        
        # Add vertical lines for percentiles if numpy was used
        if 'lengths_np' in locals():
             plt.axvline(np.percentile(lengths_np, 90), color='red', linestyle='dashed', linewidth=1, label=f'90th: {np.percentile(lengths_np, 90):.0f}')
             plt.axvline(np.percentile(lengths_np, 95), color='orange', linestyle='dashed', linewidth=1, label=f'95th: {np.percentile(lengths_np, 95):.0f}')
             plt.axvline(np.percentile(lengths_np, 99), color='purple', linestyle='dashed', linewidth=1, label=f'99th: {np.percentile(lengths_np, 99):.0f}')
             plt.legend()

        plot_save_path = Path.cwd() / "outputs" / "token_length_histogram.png"
        try:
            plt.savefig(plot_save_path)
            logger.info(f"Histogram plot saved to: {plot_save_path}")
        except Exception as e:
            logger.error(f"Could not save histogram plot: {e}")
        # plt.show() # Uncomment if running in an environment that supports GUI pop-ups
        plt.close() # Close the figure
    else:
        logger.info("Text-based histogram / frequency count:")
        counts = collections.Counter(all_token_lengths)
        # Binning for text display
        max_val = max(all_token_lengths) if all_token_lengths else 0
        bin_size = max(1, (max_val // 20) if max_val > 20 else 1) # Aim for ~20 bins
        bins = range(0, max_val + bin_size, bin_size)
        
        binned_counts = collections.defaultdict(int)
        for length in all_token_lengths:
            for i in range(len(bins) - 1):
                if bins[i] <= length < bins[i+1]:
                    binned_counts[f"{bins[i]}-{bins[i+1]-1}"] += 1
                    break
            else: # handle last bin edge case
                 if length == bins[-1]:
                     binned_counts[f"{bins[-2]}-{bins[-1]}"] +=1


        logger.info("Token Length Bins | Count")
        logger.info("-----------------|-------")
        for bin_range, count in sorted(binned_counts.items(), key=lambda item: int(item[0].split('-')[0])):
            logger.info(f"{bin_range:<17} | {count}")


def main():
    # --- Configuration ---
    jsonl_directory_path = Path.cwd() / "data" / "dcn_jsonl"
    cache_directory_path = Path.cwd() / "data" / "log_dataset_item_cache" # Cache for the index
    
    base_tokenizer_model = "EleutherAI/gpt-neox-20b" 
    # base_tokenizer_model = "state-spaces/mamba-2.8b-slimpj" # If you have this downloaded or network access
    
    batch_size_for_test = 2
    max_seq_len_for_item = 4096 # Max length for a single serialized log line after tokenization

    # --- 1. Initialize LogTokenizer ---
    logger.info(f"--- Initializing LogTokenizer with base: {base_tokenizer_model} ---")
    try:
        log_tokenizer = LogTokenizer(base_tokenizer_name=base_tokenizer_model)
    except Exception as e:
        logger.error(f"Failed to initialize LogTokenizer: {e}")
        return
    
    # find_max_context_length_for_logs(
    #     jsonl_dir=jsonl_directory_path,
    #     log_tokenizer=log_tokenizer,
    #     file_limit=None # Set to a number to limit files, e.g., 5 for testing
    # )
    
    # analyze_and_plot_token_length_distribution(
    #     jsonl_dir=jsonl_directory_path,
    #     log_tokenizer=log_tokenizer,
    #     file_limit=None, # Set to a number to limit files, e.g., 5 for testing
    #     max_token_len_for_hist=15000 # Cap histogram display range
    # ) 
    
    # --- 2. Initialize LogDataset ---
    logger.info(f"\n--- Initializing LogDataset ---")
    try:
        log_dataset = LogDataset(
            log_tokenizer=log_tokenizer,
            jsonl_dir=jsonl_directory_path,
            cache_dir=cache_directory_path,
            rebuild_index_cache=True, # Set to True for the first run or if files change
            file_limit=None, # Process all files
            max_seq_len=max_seq_len_for_item
        )
    except Exception as e:
        logger.error(f"Failed to initialize LogDataset: {e}")
        return

    logger.info(f"LogDataset initialized. Total number of log entries (lines): {len(log_dataset)}")

    # --- 3. Test with DataLoader ---
    if len(log_dataset) > 0:
        # Create the collate_fn with the specific pad_token_id from our tokenizer
        collate_function = lambda batch: collate_fn(batch, pad_token_id=log_tokenizer.tokenizer.pad_token_id)
        
        data_loader = DataLoader(log_dataset, batch_size=batch_size_for_test, shuffle=False, collate_fn=collate_function)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info("\n--- Iterating through a few batches ---")
        for i, batch in enumerate(data_loader):
            if i >= 3: # Show details for first 3 batches
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device) # Added attention mask
            
            logger.info(f"\nBatch {i+1}:")
            logger.info(f"  Input IDs shape: {input_ids.shape}, Device: {input_ids.device}")
            logger.info(f"  Labels shape: {labels.shape}, Device: {labels.device}")
            logger.info(f"  Attention Mask shape: {attention_mask.shape}, Device: {attention_mask.device}")
            
            if input_ids.numel() > 0:
                logger.info(f"  First Input ID sequence (from batch): {input_ids[0].tolist()[:30]}...")
                logger.info(f"  First Label sequence (from batch):  {labels[0].tolist()[:30]}...")
                logger.info(f"  First Attention Mask (from batch):  {attention_mask[0].tolist()[:30]}...")
                
                # logger.info(f"  Decoded Input (first item): {log_tokenizer.decode(input_ids[0][attention_mask[0] == 1].cpu().tolist())}")
                # For labels, be careful with -100 when decoding for verification
                # valid_labels = labels[0][labels[0] != -100].cpu().tolist()
                # logger.info(f"  Decoded Labels (first item, valid tokens): {log_tokenizer.decode(valid_labels)}")

    else:
        logger.warning("Dataset is empty, cannot iterate through DataLoader.")

    logger.info("\n--- Main test function finished ---")

if __name__ == "__main__":
    main()
    