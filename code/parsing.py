import re
import os
import pickle
from pathlib import Path

class LogParser:
    """
    A class to parse Warrior framework log files, extracting structured data 
    in a hierarchical manner from projects down to individual commands,
    populating detailed parent information at each level.
    """

    def __init__(self, log_file_path: str):
        """
        Initializes the LogParser with the path to the log file.
        
        Args:
            log_file_path (str): The full path to the log file.
        
        Raises:
            FileNotFoundError: If the log file does not exist.
        """
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"Log file not found at: {log_file_path}")
        self.log_file_path = log_file_path
        self.log_content = self._read_log_file()
        print(f"Log file '{log_file_path}' loaded successfully.")

    def _read_log_file(self) -> str:
        """Reads the entire log file into a single string."""
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _find_block(self, content: str, start_key: str, end_key: str) -> str:
        """Finds a single block of text between two keys, including key lines."""
        try:
            start_index = content.find(start_key)
            start_line_index = content.rfind('\n', 0, start_index) + 1
            
            end_index = content.find(end_key, start_line_index)
            end_line_index = content.find('\n', end_index)
            
            if start_index == -1 or end_index == -1:
                return ""
            return content[start_line_index:end_line_index]
        except Exception:
            return ""

    def _find_all_blocks(self, content: str, start_pattern: str, end_pattern: str) -> list:
        """A generic helper to find all non-overlapping blocks of text using regex."""
        pattern = re.compile(f"({start_pattern}.*?{end_pattern})", re.DOTALL)
        matches = pattern.finditer(content)
        return [match.group(1) for match in matches]

    def _step1_extract_project_data(self) -> dict:
        """Step 1: Extracts the main project log and the summary section."""
        print("Step 1: Extracting project data...")
        project_log = self._find_block(self.log_content, 
                                       'Executing suites sequentially', 
                                       'Project execution completed')
                                       
        project_summary = self._find_block(self.log_content, 
                                           'Execution Summary', 
                                           '++++ Results Summary ++++')
        
        project_dict = {
            "project": project_log,
            "project_summary": project_summary
        }
        
        print(f"  - Project log block found ({len(project_dict['project'])} chars).")
        print(f"  - Project summary block found ({len(project_dict['project_summary'])} chars).")
        return project_dict

    def _step2_extract_testsuites(self, project_log: str) -> list:
        """Step 2: Extracts all test suites from the project log."""
        print("Step 2: Extracting test suites...")
        testsuites_list = []
        testsuite_blocks = self._find_all_blocks(project_log, r'<<<< Starting execution of Test suite:', r'END OF TEST SUITE')

        for block in testsuite_blocks:
            name_match = re.search(r"Executing testsuite '([^']*)'", block)
            title_match = re.search(r"-I- Title: (.*)", block)
            
            testsuites_list.append({
                "name": name_match.group(1) if name_match else "N/A",
                "title": title_match.group(1).strip() if title_match else "N/A",
                "testsuite": block
            })
        print(f"  - Found {len(testsuites_list)} test suite(s).")
        return testsuites_list

    def _step3_extract_testcases(self, testsuites_list: list) -> list:
        """Step 3: Extracts all test cases from the list of test suites."""
        print("Step 3: Extracting test cases...")
        testcases_list = []
        for suite in testsuites_list:
            testcase_blocks = self._find_all_blocks(suite['testsuite'], r'<<<< Starting execution of Test case:', r'END OF TESTCASE')
            
            if not testcase_blocks:
                print(f"  - Info: No test cases found in suite '{suite['name']}'. It may have been skipped.")
                continue

            for block in testcase_blocks:
                name_match = re.search(r'<<<< Starting execution of Test case: .*/([^/]+\.xml)', block)
                title_match = re.search(r'-I- =+  TC-DETAILS  =+(?:.|\n)*?-I- Title: (.*)', block)

                testcases_list.append({
                    "parent_suite_name": suite['name'],
                    "parent_suite_title": suite['title'],
                    "name": os.path.splitext(name_match.group(1))[0] if name_match else "N/A",
                    "title": title_match.group(1).strip() if title_match else "N/A",
                    "testcase": block
                })

        print(f"  - Found {len(testcases_list)} test case(s) in total.")
        return testcases_list

    def _step4_extract_teststeps(self, testcases_list: list) -> list:
        """Step 4: Extracts all test steps from the list of test cases."""
        print("Step 4: Extracting test steps...")
        teststeps_list = []
        
        for case in testcases_list:
            start_markers = list(re.finditer(r'-I- \*{21} Keyword: (.*?) \*{21}', case['testcase']))
            
            for i, start_match in enumerate(start_markers):
                step_start_pos = start_match.start()
                next_line_start = case['testcase'].find('\n', start_match.end()) + 1
                next_line_end = case['testcase'].find('\n', next_line_start)
                next_line = case['testcase'][next_line_start:next_line_end]

                if "STATUS:SKIPPED" in next_line:
                    print(f"  - Skipping test step: '{start_match.group(1).strip()}' was skipped.")
                    continue
                
                next_step_start_pos = start_markers[i+1].start() if i + 1 < len(start_markers) else len(case['testcase'])
                block_content = case['testcase'][step_start_pos:next_step_start_pos]
                end_match = re.search(r'Keyword execution completed', block_content)
                
                if not end_match: continue
                block = block_content[:end_match.end()]

                step_num_match = re.search(r'-I- step number: (\d+)', block)
                title_match = re.search(r'-I- Teststep Description: (.*)', block)

                teststeps_list.append({
                    "parent_suite_name": case['parent_suite_name'],
                    "parent_suite_title": case['parent_suite_title'],
                    "parent_case_name": case['name'],
                    "parent_case_title": case['title'],
                    "title": title_match.group(1).strip() if title_match else "N/A",
                    "step_number": step_num_match.group(1) if step_num_match else "N/A",
                    "name": start_match.group(1).strip(),
                    "teststep": block
                })

        print(f"  - Found {len(teststeps_list)} valid test step(s) in total.")
        return teststeps_list

    def _step5_extract_substeps(self, teststeps_list: list) -> list:
        """Step 5: Extracts all substeps from the list of test steps."""
        print("Step 5: Extracting sub-steps...")
        substeps_list = []
        start_pattern = r'<< Substep >>'
        end_pattern = r'<< Substep status >>\n.*?-I-\s+STATUS:(?:PASS|FAIL|ERROR|INFO|WARN)'

        for step in teststeps_list:
            if start_pattern not in step['teststep']:
                continue
                
            substep_blocks = self._find_all_blocks(step['teststep'], start_pattern, end_pattern)
            for block in substep_blocks:
                title_match = re.search(r'-I- Keyword Description: (.*)', block)
                
                desc_dict = {
                    "testsuite_name": step['parent_suite_name'],
                    "testsuite_title": step['parent_suite_title'],
                    "testcase_name": step['parent_case_name'],
                    "testcase_title": step['parent_case_title'],
                    "teststep_name": step['name'],
                    "teststep_title": step['title'],
                    "teststep_number": step['step_number'],
                }
                
                substeps_list.append({
                    "desc": desc_dict,
                    "title": title_match.group(1).strip() if title_match else "N/A",
                    "substep": block
                })

        print(f"  - Found {len(substeps_list)} sub-step(s) in total.")
        return substeps_list

    def _step6_extract_commands(self, substeps_list: list) -> list:
        """Step 6: Extracts all commands from the list of sub-steps."""
        print("Step 6: Extracting commands...")
        commands_list = []
        start_pattern = r'-D- >>>'
        end_pattern = r'-D- <<<'

        for substep in substeps_list:
            if start_pattern.strip() not in substep['substep']:
                continue

            command_blocks = self._find_all_blocks(substep['substep'], start_pattern, end_pattern)
            for block in command_blocks:
                status = "N/A"
                for line in reversed(block.strip().split('\n')):
                    status_match = re.search(r'-I- COMMAND STATUS:(.*)', line)
                    if status_match:
                        status = status_match.group(1).strip()
                        break
                
                desc = substep['desc'].copy()
                desc["substep_title"] = substep['title']

                commands_list.append({
                    "desc": desc,
                    "command": block.strip(),
                    "status": status
                })
        
        print(f"  - Found {len(commands_list)} command(s) in total.")
        return commands_list
        
    def parse_and_save(self, output_filename="log_data.pkl"):
        """
        Orchestrates the entire parsing process from Step 1 to 6 and 
        saves the final dictionary to a specified pickle file.
        """
        print("--- Starting Full Log Parsing ---")
        
        project_dict = self._step1_extract_project_data()
        if not project_dict["project"]:
            print("Project block not found. Cannot proceed with parsing.")
            return None
            
        testsuites_list = self._step2_extract_testsuites(project_dict['project'])
        testcases_list = self._step3_extract_testcases(testsuites_list)
        teststeps_list = self._step4_extract_teststeps(testcases_list)
        substeps_list = self._step5_extract_substeps(teststeps_list)
        commands_list = self._step6_extract_commands(substeps_list)

        final_data = {
            "project_summary": project_dict['project_summary'],
            "testsuites_list": testsuites_list,
            "testcases_list": testcases_list,
            "teststeps_list": teststeps_list,
            "substeps_list": substeps_list,
            "commands_list": commands_list
        }
        
        with open(output_filename, 'wb') as f:
            pickle.dump(final_data, f)
            
        print(f"\n--- Parsing Complete ---")
        print(f"Successfully saved all extracted data to '{output_filename}'")
        return final_data

# --- Main execution block ---
def main():
    # IMPORTANT: Place your log file in the same directory as this script
    # and name it 'warrior_log.txt', or provide the full path here.
    log_directory_train = Path.cwd() / Path("data/dcn_train")
    log_directory_test = Path.cwd() / Path("data/dcn_test")
    log_path_list = []
    for i in log_directory_train.iterdir():
        log_path_list.append(i)
    for i in log_directory_test.iterdir():
        log_path_list.append(i)
    
    for i in range(len(log_path_list)):
        LOG_FILE_PATH = str(log_path_list[i])
        OUTPUT_PICKLE_FILE = Path.cwd() / Path(f"outputs/parsed_logs/{log_path_list[i].stem}_parsed.pkl")

        try:
            # 1. Instantiate the parser with the log file path.
            parser = LogParser(LOG_FILE_PATH)

            # 2. Run the full parsing and saving process.
            all_data = parser.parse_and_save(output_filename=OUTPUT_PICKLE_FILE)

            # 3. Verification and summary after parsing.
            if all_data:
                print("\n--- Verification Summary ---")
                print(f"Total Test Suites Extracted: {len(all_data.get('testsuites_list', []))}")
                print(f"Total Test Cases Extracted:  {len(all_data.get('testcases_list', []))}")
                print(f"Total Test Steps Extracted:  {len(all_data.get('teststeps_list', []))}")
                print(f"Total Sub-steps Extracted:   {len(all_data.get('substeps_list', []))}")
                print(f"Total Commands Extracted:    {len(all_data.get('commands_list', []))}")
            else:
                print("Parsing did not yield any data. Please check the log file and patterns.")

            # 4. Demonstrate loading the data from the saved pickle file.
            print(f"\n--- Demonstrating Loading from '{OUTPUT_PICKLE_FILE}' ---")
            if os.path.exists(OUTPUT_PICKLE_FILE):
                with open(OUTPUT_PICKLE_FILE, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                print("Pickle file loaded successfully!")
                
                if loaded_data.get('commands_list'):
                    first_command = loaded_data['commands_list'][0]
                    print("\nSample from loaded data (first command's description):")
                    # Pretty print the description dictionary
                    for key, value in first_command['desc'].items():
                        print(f"  {key}: {value}")
                else:
                    print("Loaded data contains no commands to display.")
            else:
                print(f"Error: Output file '{OUTPUT_PICKLE_FILE}' was not created.")

        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print("Please make sure the log file exists and the LOG_FILE_PATH is correct.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during parsing: {e}")
            
        print(f"\n[{i}] --- End of Processing for this log file ---\n")
        
if __name__ == "__main__":
    main()