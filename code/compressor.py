import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Set
import torch
from sentence_transformers import CrossEncoder

class HyperGraphCompressor:
    """
    Compresses a test case's hypergraph by pruning less relevant 'passed' commands
    and retaining all 'failed' commands, preserving the final chronological order.
    """

    def __init__(self, testcase_hyperedges: List[Dict], model: CrossEncoder, device: str, k_per_failure: int = 2):
        """
        Initializes the compressor for a single test case.

        Args:
            testcase_hyperedges (List[Dict]): A list of hyperedge dictionaries for one test case.
            model (CrossEncoder): A pre-loaded CrossEncoder model.
            device (str): The device to run the model on ('cuda' or 'cpu').
            k_per_failure (int): The number of top relevant 'passed' commands to select as seeds for each failure.
        """
        self.testcase_hyperedges = testcase_hyperedges
        self.model = model
        self.device = device
        self.k_per_failure = k_per_failure
        self.passed_hyperedges = []
        self.failed_hyperedges = []
        
        self._group_and_sort_commands()

    def _group_and_sort_commands(self):
        """
        Partitions commands into 'passed' and 'failed' groups and gives them an
        original index to track them uniquely.
        """
        for i, cmd in enumerate(self.testcase_hyperedges):
            cmd['_original_index'] = i 
            status = cmd.get("hyperedge", {}).get("final_status", "").upper()
            if status == "PASS":
                self.passed_hyperedges.append(cmd)
            elif status in ["FAIL", "ERROR"]:
                self.failed_hyperedges.append(cmd)
        
        date_format = "%Y-%m-%d %H:%M:%S"
        # Sorting the original lists is not strictly needed for logic but good for ordered processing
        self.passed_hyperedges.sort(key=lambda x: datetime.strptime(x['start_timestamp'], date_format) if x['start_timestamp'] != "N/A" else datetime.min)
        self.failed_hyperedges.sort(key=lambda x: datetime.strptime(x['start_timestamp'], date_format) if x['start_timestamp'] != "N/A" else datetime.min)

    def _get_causal_passed_commands(self, failed_cmd_timestamp_str: str) -> List[Dict]:
        """
        Returns a list of 'passed' commands that occurred before a given timestamp.
        """
        if failed_cmd_timestamp_str == "N/A": return []
        date_format = "%Y-%m-%d %H:%M:%S"
        failed_time = datetime.strptime(failed_cmd_timestamp_str, date_format)
        return [p_cmd for p_cmd in self.passed_hyperedges if p_cmd['start_timestamp'] != "N/A" and datetime.strptime(p_cmd['start_timestamp'], date_format) < failed_time]

    @staticmethod
    def _hyperedge_to_text_nodes(hyperedge: Dict) -> List[str]:
        """Converts hyperedge attribute values into a flat list of strings."""
        if not hyperedge or not isinstance(hyperedge, dict): return []
        return [str(value) for value in hyperedge.values() if value is not None]

    def compress(self) -> List[Dict]:
        """
        Orchestrates the compression process: identifies seed nodes from passed commands,
        retains all failed commands, and re-sorts the combined result chronologically.

        Returns:
            A list of command dictionaries representing the compressed test case.
        """
        print(f"  - Original command counts: {len(self.passed_hyperedges)} passed, {len(self.failed_hyperedges)} failed.")

        if not self.failed_hyperedges or not self.passed_hyperedges:
            print("  - WARNING: No failed or passed commands exist to perform causal analysis. Returning original uncompressed sequence.")
            return self.testcase_hyperedges

        # Use a set to store the original indices of selected seed nodes to avoid duplicates
        unique_seed_indices: Set[int] = set()

        for i, failed_cmd in enumerate(self.failed_hyperedges):
            print(f"    - Analyzing relevance for failure {i+1}/{len(self.failed_hyperedges)}...")
            
            causal_passed_cmds = self._get_causal_passed_commands(failed_cmd['start_timestamp'])
            if not causal_passed_cmds:
                print("      - No preceding passed commands found for this failure. Skipping.")
                continue

            failed_nodes = self._hyperedge_to_text_nodes(failed_cmd.get('hyperedge', {}))
            scores_for_this_failure = {}

            all_pairs = []
            cmd_pair_indices = []
            for passed_cmd in causal_passed_cmds:
                passed_nodes = self._hyperedge_to_text_nodes(passed_cmd.get('hyperedge', {}))
                pairs = [[fn, pn] for fn in failed_nodes for pn in passed_nodes]
                if pairs:
                    all_pairs.extend(pairs)
                    cmd_pair_indices.extend([passed_cmd['_original_index']] * len(pairs))
            
            if not all_pairs: continue

            similarity_scores = self.model.predict(all_pairs, show_progress_bar=False)
            
            for idx, score in zip(cmd_pair_indices, similarity_scores):
                if idx not in scores_for_this_failure: scores_for_this_failure[idx] = []
                scores_for_this_failure[idx].append(score)

            mean_scores = {idx: np.mean(scores) for idx, scores in scores_for_this_failure.items()}
            sorted_by_relevance = sorted(mean_scores, key=mean_scores.get, reverse=True)
            top_k_indices = sorted_by_relevance[:self.k_per_failure]
            
            unique_seed_indices.update(top_k_indices)

        print(f"\n  - Total unique passed commands to retain: {len(unique_seed_indices)}")

        # Retrieve the full command objects for the retained passed commands
        retained_passed_hyperedges = [self.testcase_hyperedges[i] for i in unique_seed_indices]

        # The final compressed testcase is the combination of ALL failed and the RETAINED passed hyperedges
        compressed_testcase = self.failed_hyperedges + retained_passed_hyperedges
        
        # Re-sort the final combined list chronologically to maintain the original sequence
        date_format = "%Y-%m-%d %H:%M:%S"
        compressed_testcase.sort(key=lambda x: datetime.strptime(x['start_timestamp'], date_format) if x['start_timestamp'] != "N/A" else datetime.min)

        print(f"  - Compressed test case from {len(self.testcase_hyperedges)} to {len(compressed_testcase)} commands.")
        
        return compressed_testcase


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    HYPERGRAPH_INPUT_PICKLE = "outputs/hypergraphs/hypergraph_data.pkl"
    COMPRESSED_OUTPUT_PICKLE = "outputs/comp_hg_CE/compressed_hypergraph_data.pkl"
    K_PER_FAILURE = 2 # Select the top 2 most relevant preceding passed commands for each failure.
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    # --- 1. Setup Model and Device ---
    print("--- Initializing Cross-Encoder Model ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512, device=device)

    # --- 2. Load Pre-processed Hypergraph Data ---
    print(f"\n--- Loading hypergraph data from '{HYPERGRAPH_INPUT_PICKLE}' ---")
    if not os.path.exists(HYPERGRAPH_INPUT_PICKLE):
        print(f"ERROR: Input file '{HYPERGRAPH_INPUT_PICKLE}' not found. Run the previous script first.")
    else:
        with open(HYPERGRAPH_INPUT_PICKLE, 'rb') as f:
            hypergraph_collection = pickle.load(f)
        
        print(f"Loaded hypergraph data for {len(hypergraph_collection)} testcases.")
        
        # --- 3. Process Each Testcase to Create a Compressed Hypergraph ---
        compressed_hypergraph_collection = {}
        print("\n--- Starting Hypergraph Compression Process ---")
        total_testcases = len(hypergraph_collection)

        for i, (tc_name, hyperedges) in enumerate(hypergraph_collection.items()):
            print(f"\nProcessing Testcase {i+1}/{total_testcases}: '{tc_name}'")
            
            # Using the corrected seed node identification logic
            compressor = HyperGraphCompressor(
                testcase_hyperedges=hyperedges, 
                model=cross_encoder,
                device=device,
                k_per_failure=K_PER_FAILURE
            )
            
            # The compress method now returns the final chronologically sorted list
            compressed_testcase = compressor.compress()
            compressed_hypergraph_collection[tc_name] = compressed_testcase

        # --- 4. Save the Final Compressed Data ---
        print(f"\n--- Saving compressed hypergraph data to '{COMPRESSED_OUTPUT_PICKLE}' ---")
        with open(COMPRESSED_OUTPUT_PICKLE, 'wb') as f:
            pickle.dump(compressed_hypergraph_collection, f)
        print("Save complete.")

        # --- 5. Verification ---
        print("\n--- Verification of Saved Compressed Data ---")
        with open(COMPRESSED_OUTPUT_PICKLE, 'rb') as f:
            loaded_compressed_data = pickle.load(f)
            
        if loaded_compressed_data:
            first_testcase_key = list(loaded_compressed_data.keys())[0]
            original_tc = hypergraph_collection[first_testcase_key]
            compressed_tc = loaded_compressed_data[first_testcase_key]
            
            print(f"Verification for testcase '{first_testcase_key}':")
            print(f"  - Original command count: {len(original_tc)}")
            print(f"  - Compressed command count: {len(compressed_tc)}")
            
            timestamps = [datetime.strptime(cmd['start_timestamp'], "%Y-%m-%d %H:%M:%S") for cmd in compressed_tc if cmd['start_timestamp'] != 'N/A']
            is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            print(f"  - Chronological sequence maintained: {is_sorted}")
            
            original_failed_count = sum(1 for cmd in original_tc if cmd['hyperedge']['final_status'] in ['FAIL', 'ERROR'])
            compressed_failed_count = sum(1 for cmd in compressed_tc if cmd['hyperedge']['final_status'] in ['FAIL', 'ERROR'])
            print(f"  - All failed commands retained: {original_failed_count == compressed_failed_count} ({compressed_failed_count} of {original_failed_count})")
            
            print("\n  - First 5 commands in the compressed sequence:")
            for cmd in compressed_tc[:5]:
                print(f"    - Timestamp: {cmd['start_timestamp']}, Status: {cmd['hyperedge']['final_status']}, Command: {cmd['hyperedge']['command_string'][:40]}...")
        else:
            print("No data was compressed in the final output.")