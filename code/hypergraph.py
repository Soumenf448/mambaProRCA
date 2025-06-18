import os
import re
import pickle
from typing import List, Optional, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_vertexai import VertexAI
import os
import google.oauth2.credentials
from pathlib import Path

# --- Configuration ---
# Load environment variables from the specified file
load_dotenv('code/api.env')
credentials = google.oauth2.credentials.Credentials(os.environ["GOOGLE_GEMINI_API_KEY"])
api_endpoint = "https://api.ai-service.global.fujitsu.com/ai-foundation/chat-ai/gemini/flash:generateContent"
    
# --- Pydantic Model for Structured LLM Output ---
# This defines the schema for a single hyperedge (one command).

class CommandAttributes(BaseModel):
    """Defines the structured attributes to be extracted from a raw command log."""
    command_string: str = Field(description="The exact, clean command that was executed.")
    target_system: str = Field(description="The system or device where the command was executed (e.g., NE2, NE3_TRIB1_DIP).")
    final_status: str = Field(description="The final outcome of the command block (PASS, FAIL, ERROR).")
    command_type: str = Field(description="Categorize the command's purpose (e.g., Configuration, Verification, System_Interaction, File_System, Other).")
    key_event_summary: str = Field(description="A one-sentence summary of the most important event in the command's response (e.g., 'Commit complete.', 'Found a match for <dcnMode>.', 'Element does not exist.').")
    initial_execution_timestamp: str = Field(description="The UTC timestamp of the first 'Sending Command:' log line, in YYYY-MM-DDTHH:MM:SSZ format.")
    final_attempt_duration_sec: float = Field(description="The duration in seconds of the FINAL execution attempt.")
    is_retried: bool = Field(description="true if the log contains 'RETRIAL ATTEMPT', otherwise false.")
    verification_logic: str = Field(description="A concise summary of the verification rule that was applied (e.g., \"Found 'ok' in response.\").")
    failure_info: Optional[str] = Field(description="If final_status is FAIL/ERROR, the specific error message; otherwise null.")

    @field_validator("command_type")
    def validate_command_type(cls, value):
        valid_types = {"Configuration", "Verification", "System_Interaction", "File_System", "Other"}
        if value not in valid_types:
            raise ValueError(f"command_type must be one of {valid_types}")
        return value

# --- Helper Function to Pre-process Command List ---

def group_commands_by_testcase(commands_list: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Groups the flat list of commands into a dictionary keyed by testcase_name.
    """
    grouped_commands = {}
    for command in commands_list:
        tc_name = command.get('desc', {}).get('testcase_name', 'Unknown_TestCase')
        if tc_name not in grouped_commands:
            grouped_commands[tc_name] = []
        grouped_commands[tc_name].append(command)
    return grouped_commands


# --- Main Class for Hypergraph Creation ---

class HyperGraph:
    """
    Generates structured hyperedges (command attributes) from raw log snippets
    using a powerful LLM like Gemini.
    """
    def __init__(self, llm_instance: VertexAI):
        self.llm = llm_instance
        self.parser = PydanticOutputParser(pydantic_object=CommandAttributes)
        self.prompt_template = self._create_prompt_template()
        self.chain = self.prompt_template | self.llm | self.parser

    def _create_prompt_template(self) -> PromptTemplate:
        """Creates the detailed zero-shot prompt for Gemini."""
        
        template = """
# TASK: LOG COMMAND ANALYSIS AND STRUCTURED ATTRIBUTE EXTRACTION

You are an expert Log Analysis Engine for the Warrior Test Framework. Your task is to analyze the provided raw log snippet for a single command execution and extract a structured set of 10 key-value attributes that captures the entire command in detail.

### CONTEXT OF THE COMMAND
- **Test Suite:** {desc_testsuite_name} ({desc_testsuite_title})
- **Test Case:** {desc_testcase_name} ({desc_testcase_title})
- **Test Step:** {desc_teststep_number} - {desc_teststep_name} ({desc_teststep_title})
- **Sub-step:** {desc_substep_title}

### RAW COMMAND LOG SNIPPET
```log
{command}
```

### INSTRUCTIONS
Analyze the raw log snippet and extract the specified attributes. The output MUST be a valid JSON object matching the format below.
**IMPORTANT:** If the log shows multiple retry attempts, all attributes must reflect the FINAL, decisive state. For example, `final_attempt_duration_sec` should be the duration of the *last* attempt only.

{format_instructions}
"""
        return PromptTemplate(
            input_variables=["desc_testsuite_name", "desc_testsuite_title", "desc_testcase_name", "desc_testcase_title",
                             "desc_teststep_number", "desc_teststep_name", "desc_teststep_title",
                             "desc_substep_title", "command"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template=template
        )

    def create_hyperedges_for_testcase(self, testcase_commands: List[Dict]) -> List[Dict]:
        """
        Processes a list of commands for a single testcase and returns a list of hyperedges.

        Args:
            testcase_commands (List[Dict]): A list of command objects from the parser.

        Returns:
            List[Dict]: A list where each item is a dictionary representing one command's hyperedge.
        """
        hyperedges_list = []
        for i, command_data in enumerate(testcase_commands):
            print(f"  - Processing command {i+1}/{len(testcase_commands)} for testcase '{command_data['desc']['testcase_name']}'...")
            
            raw_command = command_data.get('command', '')
            desc = command_data.get('desc', {})
            
            # Extract basic info directly
            testcase_name = desc.get('testcase_name', 'N/A')
            command_status = command_data.get('status', 'N/A')
            
            # Extract timestamps using regex
            start_ts_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Sending Command:', raw_command)
            end_ts_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Command completed', raw_command)
            start_timestamp = start_ts_match.group(1) if start_ts_match else "N/A"
            end_timestamp = end_ts_match.group(1) if end_ts_match else "N/A"

            try:
                # Invoke the LLM chain to get structured attributes
                hyperedge_attributes = self.chain.invoke({
                    "desc_testsuite_name": desc.get('testsuite_name', 'N/A'),
                    "desc_testsuite_title": desc.get('testsuite_title', 'N/A'),
                    "desc_testcase_name": testcase_name,    
                    "desc_testcase_title": desc.get('testcase_title', 'N/A'),
                    "desc_teststep_number": desc.get('teststep_number', 'N/A'),
                    "desc_teststep_name": desc.get('teststep_name', 'N/A'),
                    "desc_teststep_title": desc.get('teststep_title', 'N/A'),
                    "desc_substep_title": desc.get('substep_title', 'N/A'), 
                    "command": raw_command
                })
                
                hyperedge_dict = {
                    "testcase_name": testcase_name,
                    "command_status": command_status,
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "raw_command": raw_command,
                    "hyperedge": hyperedge_attributes.model_dump() # Convert Pydantic model to dict
                }
                hyperedges_list.append(hyperedge_dict)

            except Exception as e:
                print(f"    - WARNING: Failed to parse LLM output for command {i+1}. Error: {e}. Skipping this command.")
                # Optionally, append an error entry
                hyperedges_list.append({
                    "testcase_name": testcase_name,
                    "command_status": "PARSING_ERROR",
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "raw_command": raw_command,
                    "hyperedge": {"error": str(e)}
                })
                
        return hyperedges_list


# --- Main Execution Block ---
def run_single_testcase(PARSED_LOG_PICKLE, HYPERGRAPH_OUTPUT_PICKLE):
    # --- 1. Load and Pre-process Data ---
    print(f"--- Loading parsed data from '{PARSED_LOG_PICKLE}' ---")
    if not os.path.exists(PARSED_LOG_PICKLE):
        print(f"ERROR: Input file '{PARSED_LOG_PICKLE}' not found. Please run the previous parsing script first.")
    else:
        with open(PARSED_LOG_PICKLE, 'rb') as f:
            loaded_data = pickle.load(f)
        
        commands_list = loaded_data.get("commands_list", [])
        if not commands_list:
            print("No commands found in the loaded data. Exiting.")
        else:
            print(f"Loaded {len(commands_list)} commands.")
            
            # Group commands by testcase name for processing
            grouped_by_testcase = group_commands_by_testcase(commands_list)
            print(f"Grouped commands into {len(grouped_by_testcase)} unique testcases.")

            # --- 2. Initialize LLM and HyperGraph Builder ---
            print("\n--- Initializing Gemini LLM and HyperGraph Builder ---")
            llm = VertexAI(
                model="gemini-2.0-flash-001",
                temperature=1,
                max_tokens=None,
                max_retries=6,
                stop=None,
                credentials=credentials,
                project="dummy",
                streaming=False,
                api_transport="rest",
                api_endpoint=api_endpoint,
            )
                    
            hypergraph_builder = HyperGraph(llm_instance=llm)

            # --- 3. Process Each Testcase to Create Hyperedges ---
            final_hypergraph_collection = {}
            total_testcases = len(grouped_by_testcase)
            
            print("\n--- Starting Hyperedge Creation Process ---")
            for idx, (tc_name, tc_commands) in enumerate(grouped_by_testcase.items()):
                print(f"\nProcessing Testcase {idx + 1}/{total_testcases}: '{tc_name}'")
                
                # Create hyperedges for all commands in the current testcase
                hyperedges = hypergraph_builder.create_hyperedges_for_testcase(tc_commands)
                
                # Store the list of hyperedges under the testcase name key
                final_hypergraph_collection[tc_name] = hyperedges
                
            # --- 4. Save the Final Hypergraph Data ---
            print(f"\n--- Saving final hypergraph data to '{HYPERGRAPH_OUTPUT_PICKLE}' ---")
            with open(HYPERGRAPH_OUTPUT_PICKLE, 'wb') as f:
                pickle.dump(final_hypergraph_collection, f)
            print("Save complete.")

            # --- 5. Verification ---
            print("\n--- Verification of Saved Data ---")
            with open(HYPERGRAPH_OUTPUT_PICKLE, 'rb') as f:
                final_data = pickle.load(f)
            
            first_testcase_key = list(final_data.keys())[0]
            first_hyperedge = final_data[first_testcase_key][0]
            
            print(f"Sample hyperedge from testcase '{first_testcase_key}':")
            print(f"  Command Status: {first_hyperedge['command_status']}")
            print(f"  Raw Command Snippet Length: {len(first_hyperedge['raw_command'])} chars")
            print("  Extracted Hyperedge Attributes (from Gemini):")
            for key, val in first_hyperedge['hyperedge'].items():
                print(f"    - {key}: {val}")
                
# --- Main Execution Block ---
def main():
    # Define input and output file paths
    parsed_file_dir = Path.cwd() / Path("outputs/parsed_logs")
    for i in parsed_file_dir.iterdir():  
        PARSED_LOG_PICKLE = i
        HYPERGRAPH_OUTPUT_PICKLE = Path.cwd() / Path(f"outputs/hypergraphs/{i.stem}_hypergraph.pkl")
        run_single_testcase(PARSED_LOG_PICKLE, HYPERGRAPH_OUTPUT_PICKLE)
        print(f"\n[{i}] --- End of Processing for this log file ---\n")
        
if __name__ == "__main__":
    main()