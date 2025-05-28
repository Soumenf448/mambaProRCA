import re
import json
from pathlib import Path
from datetime import datetime, timezone
import csv
import logging
from tqdm import tqdm

# --- Standard Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("LogPreprocessor")
logger.setLevel(logging.INFO) # Keep it to INFO to avoid debug noise unless specifically needed
# --- End Logging Setup ---

class LogPreprocessor:
    def __init__(self, data_directory: str | Path = "data/dcn"):
        self.current_dir = Path.cwd()
        self.data_directory = (self.current_dir / Path(data_directory)).resolve()
        self.output_base_dir = self.data_directory.parent / f"{self.data_directory.name}_jsonl"
        self.index_file_path = self.data_directory / "log_file_index.csv"
        self.use_index_cache = True

        self._timestamp_patterns = [
            re.compile(r'\[?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[,\.]\d{1,9}(?:Z|[+-]\d{2}:\d{2})?)\]?'),
            re.compile(r'(\d{4}[/-]\d{2}[/-]\d{2}\s\d{2}:\d{2}:\d{2}[,\.]\d{1,6})'),
            re.compile(r'(\d{4}[/-]\d{2}[/-]\d{2}\s\d{2}:\d{2}:\d{2})'),
            re.compile(r'([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?\s+\d{4})'),
            re.compile(r'([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)'),
            re.compile(r'(\d{2})(\d{2})(\d{2})\s+(\d{2})(\d{2})(\d{2})'),
            re.compile(r'(\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)'),
        ]
        self._log_level_patterns = re.compile(
            r'^(INFO|ERROR|WARN|WARNING|DEBUG|TRACE|FATAL|CRITICAL|SEVERE|NOTICE|-I-|-D-|-W-|-E-|-N-)\b',
            re.IGNORECASE
        )
        self._warrior_prefix_pattern = re.compile(
            r'(\d{8}\.\d{6})\s+(stdout|stderr):([a-zA-Z0-9\._-]+)(?:\.Warrior)?\s*#\s*(.*)'
        )
        
        self._ansi_cursor_pattern = re.compile(r'^\s*\x1b\[\?25[lh]\s*$')
        
        self._detailed_progress_bar_pattern = re.compile(
            r"^\s*"
            r"(?:\x1b\[K)?\s*"
            r"(?:\d{1,3}\s*%\s*)?"
            r"\|"
            r"[█▏▎▍▌▋▊▉\u2580-\u259F\s.-]*"
            r"\|"
            r".*$"
        )
        # Modified to be more general for git-like progress, including "Resolving deltas"
        # and making "remote:" optional.
        self._git_like_progress_pattern = re.compile(
            r"^\s*(?:remote:\s*)?" # Optional "remote: "
            r"(?:Counting|Compressing|Receiving|Resolving)\s+(?:objects|deltas):\s+" # Common git operations and terms
            r"\d{1,3}%\s*\(\d+/\d+\)" # XX% (current/total)
            r"(?:,\s*\S.*?)*?" # Optional trailing info like ", done." or speed, making it non-greedy
            r"\s*(?:\x1b\[K)?\s*$" # Optional ANSI clear and end of line
        )

        self._strong_progress_indicators = re.compile(
            r"(\b\d{1,3}\s*%)"
            r"|(\b(?:[KMGT]i?B/s|bytes/s|it/s)\b)"
            r"|(\beta\s+\d{1,2}:\d{2}(?::\d{2})?\b)"
            r"|(\b\d+/\d+\s*(?:files|objects|items|tasks|Total)\b)"
            r"|(\b[\d.,]+\s*(?:[KMGT]i?B|bytes|B|KiB|MiB|GiB|TiB)\b)"
        )
        self._progress_bar_chars_pattern = re.compile(r"[|█▏▎▍▌▋▊▉\u2580-\u259F]")

        self._percentage_progress_pattern = re.compile(r'\b\d{1,3}%\s*\([\d.]+(?:kB|MB|GB|TB|KiB|MiB|GiB|TiB)?(?:/s)?\)')
        self._standalone_percentage_pattern = re.compile(r"^\s*(?:\x1b\[K)?\s*\d{1,3}%\s*(?:complete|processed|done|finished)?\s*$")

        self._curl_progress_header_pattern = re.compile(r"^\s*% Total\s+% Received % Xferd\s+Average Speed\s+Time\s+Time\s+Time\s+Current\s*$")
        self._curl_progress_stats_pattern = re.compile(r"^\s*\d+\s+\d+(?:\.\d+)?[KMGT]?\s+\d+\s+\d+(?:\.\d+)?[KMGT]?\s+\d+\s+\d+(?:\.\d+)?[KMGT]?\s+\d+(?:\.\d+)?[KMGTpb]?\s+(?:(?:[\d:]+)|(?:--:--:--)){3}\s+\d+(?:\.\d+)?[KMGTpb]?\s*$")
        
        self._project_pattern = re.compile(r"^-I- Project:(\S+)\s+STATUS:(\w+)")
        self._testsuite_pattern = re.compile(r"^-I- Testsuite:(\S+)\s+STATUS:(\w+)")
        self._testcase_pattern = re.compile(r"^-I- TESTCASE:(\S+)\s+STATUS:(\w+)")
        self._keyword_status_pattern = re.compile(r"^-I- KEYWORD:(\S+)\s+STATUS:(\w+)")
        self._command_status_pattern = re.compile(r"^-I- COMMAND STATUS:(\w+)")
        self._step_number_pattern = re.compile(r"^-I- step number:\s*(\d+)")
        self._step_desc_pattern = re.compile(r"^-I- Teststep Description:\s*(.+)")
        self._keyword_start_pattern = re.compile(r"^-I- \*{5,} Keyword: (\S+) \*{5,}")
        self._testsuite_start_pattern = re.compile(r"^-I- \${5,} START OF TEST SUITE : (\S+) \${5,}")
        self._testcase_start_pattern = re.compile(r"^-I- ={5,} START OF TESTCASE : (\S+) ={5,}")

    def _discover_log_files(self) -> list[Path]:
        log_files = []
        if not self.data_directory.exists():
            logger.error(f"Data directory does not exist: {self.data_directory}")
            return log_files
        for p in self.data_directory.rglob('*'):
            if p.is_file() and (p.suffix.lower() in ['.log', '.txt'] or not p.suffix):
                log_files.append(p)
        return log_files

    def manage_index(self) -> list[Path]:
        log_files_to_process = []
        rebuild_needed = False
        if self.index_file_path.exists():
            if not self.use_index_cache:
                logger.info("Index file exists, but use_index_cache is False. Rebuilding index.")
                rebuild_needed = True
            else:
                logger.info("Index file exists and use_index_cache is True. Using existing index file.")
        else:
            logger.info("No index file found. Building new index.")
            rebuild_needed = True

        if rebuild_needed:
            self._build_index()

        try:
            with open(self.index_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                if 'file_path' not in (reader.fieldnames or []):
                    logger.error("Index file is missing 'file_path' column or is empty. Rebuilding.")
                    self._build_index()
                    with open(self.index_file_path, 'r', newline='', encoding='utf-8') as rebuilt_csvfile:
                        reader = csv.DictReader(rebuilt_csvfile)
                        if 'file_path' not in (reader.fieldnames or []):
                            logger.error("Rebuilt index file still malformed. Cannot proceed.")
                            return []
                for row in reader:
                    log_files_to_process.append(Path(row['file_path']))
        except FileNotFoundError:
            logger.error(f"Index file {self.index_file_path} not found even after build attempt. Critical error.")
            return []
        except Exception as e:
            logger.error(f"Error reading index file {self.index_file_path}: {e}. Attempting to rebuild once more.")
            self._build_index()
            try:
                with open(self.index_file_path, 'r', newline='', encoding='utf-8') as rebuilt_csvfile:
                    reader = csv.DictReader(rebuilt_csvfile)
                    if 'file_path' not in (reader.fieldnames or []):
                        logger.error("Rebuilt index file still malformed. Cannot proceed.")
                        return []
                    for row in reader:
                        log_files_to_process.append(Path(row['file_path']))
            except Exception as final_e:
                logger.error(f"Failed to read index file even after final rebuild: {final_e}. Cannot proceed.")
                return []
        
        if not log_files_to_process and self._discover_log_files():
            logger.warning("Index was empty but log files exist in the directory. Rebuilding index.")
            self._build_index()
            current_cache_policy = self.use_index_cache
            self.use_index_cache = True 
            result = self.manage_index()
            self.use_index_cache = current_cache_policy
            return result

        if not log_files_to_process:
            logger.warning("No files found in the index or data directory to process.")
        return log_files_to_process

    def _build_index(self):
        logger.info(f"Building index for directory: {self.data_directory}")
        discovered_files = self._discover_log_files()
        try:
            self.data_directory.mkdir(parents=True, exist_ok=True)
            with open(self.index_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['file_path']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                if discovered_files:
                    for file_path in discovered_files:
                        writer.writerow({'file_path': str(file_path.resolve())})
                    logger.info(f"Index built successfully with {len(discovered_files)} files: {self.index_file_path}")
                else:
                    logger.info(f"No files found in {self.data_directory}. Empty index created: {self.index_file_path}")
        except IOError as e:
            logger.error(f"Error writing index file {self.index_file_path}: {e}")

    def _parse_datetime_from_string(self, ts_str: str, matched_pattern: re.Pattern) -> datetime | None:
        try:
            if matched_pattern.pattern == r'(\d{2})(\d{2})(\d{2})\s+(\d{2})(\d{2})(\d{2})':
                m = matched_pattern.search(ts_str) 
                if m:
                    year_short, month, day, hour, minute, second = m.groups()
                    year = int(year_short)
                    current_year_short = datetime.now().year % 100
                    if year > current_year_short + 10: 
                        year += 1900
                    else:
                        year += 2000
                    return datetime(year, int(month), int(day), int(hour), int(minute), int(second))
            elif matched_pattern.pattern == r'(\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)':
                standardized_ts_str = ts_str.replace('.', '-', 2).replace('-', ' ', 1).replace('.', ':', 2)
                return datetime.strptime(standardized_ts_str, '%Y-%m-%d %H:%M:%S.%f')

            from dateutil import parser as dateutil_parser 
            return dateutil_parser.parse(ts_str)
        except Exception:
            return None

    def _extract_dt_and_message_from_line(self, line: str) -> tuple[datetime | None, str, str | None, dict]:
        message_part = line
        dt_obj = None
        raw_ts_str_external = None 
        warrior_context = {}

        primary_ts_match = self._timestamp_patterns[0].match(line)
        if primary_ts_match:
            raw_ts_str_external = primary_ts_match.group(1)
            dt_obj = self._parse_datetime_from_string(raw_ts_str_external, self._timestamp_patterns[0])
            if dt_obj:
                message_part = line[primary_ts_match.end():].lstrip()
        
        warrior_match = self._warrior_prefix_pattern.match(message_part)
        if warrior_match:
            warrior_ts_internal_str, stream_type, context_id, actual_message = warrior_match.groups()
            warrior_context['warrior_timestamp_internal'] = warrior_ts_internal_str
            warrior_context['warrior_stream'] = stream_type
            warrior_context['warrior_context_id'] = context_id
            parts = context_id.split('.')
            if len(parts) > 0: warrior_context['warrior_node_id'] = parts[0]
            if len(parts) > 1: warrior_context['warrior_suite_id'] = parts[1] 
            if len(parts) > 2: warrior_context['warrior_step_id'] = ".".join(parts[2:])
            message_part = actual_message
        
        if not dt_obj and not primary_ts_match: 
            for pattern_idx, pattern in enumerate(self._timestamp_patterns[1:], start=1): 
                match = pattern.match(line) 
                if match:
                    raw_ts_str_external = match.group(1) 
                    dt_obj = self._parse_datetime_from_string(raw_ts_str_external, pattern)
                    if dt_obj:
                        if not warrior_context: 
                             message_part = line[match.end():].lstrip()
                        break 
        return dt_obj, message_part.strip(), raw_ts_str_external, warrior_context

    def _normalize_timestamp_to_utc_iso(self, dt_obj: datetime | None) -> str | None:
        if not dt_obj: return None
        if dt_obj.tzinfo is None: dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        else: dt_obj = dt_obj.astimezone(timezone.utc)
        return dt_obj.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _is_line_to_be_filtered(self, line_content_to_check: str, warrior_context: dict) -> bool:
        # Rule F1: Specific ANSI codes
        if self._ansi_cursor_pattern.match(line_content_to_check):
            return True
        if line_content_to_check == "\x1b[K" or line_content_to_check == "\x1b[0m":
            return True

        # Rule F2a: Detailed progress bar pattern (|...| structure)
        if self._detailed_progress_bar_pattern.match(line_content_to_check):
            return True
        
        # Rule F2b: Git-like progress lines (e.g., remote: Counting objects: XX% (a/b))
        if self._git_like_progress_pattern.match(line_content_to_check):
            return True

        # Rule F3: Heuristic - Lines with bar characters AND strong progress indicators
        bar_char_matches = self._progress_bar_chars_pattern.findall(line_content_to_check)
        if len(bar_char_matches) >= 2: 
            if self._strong_progress_indicators.search(line_content_to_check):
                return True
        
        # Rule F4: Other percentage-based progress lines
        if self._percentage_progress_pattern.search(line_content_to_check): # e.g. 10% (10kB/s)
            return True
        if self._standalone_percentage_pattern.match(line_content_to_check): # e.g. ESC[K 75%
            return True

        # Rule F5: Curl-specific progress
        if self._curl_progress_header_pattern.match(line_content_to_check):
            return True
        if self._curl_progress_stats_pattern.match(line_content_to_check):
            return True
            
        # Rule F6: Pip download/installation messages (if not a Warrior message)
        pip_keywords = ["Collecting ", "Downloading https://", "Installing collected packages:", 
                        "Successfully installed", "Found existing installation:", "Uninstalling", 
                        "Successfully uninstalled", "You are using pip version", 
                        "You should consider upgrading via the", "Looking in indexes:"]
        if not warrior_context.get('warrior_context_id'):
            if "Downloading https://" in line_content_to_check and \
               (".whl" in line_content_to_check or ".tar.gz" in line_content_to_check or ".zip" in line_content_to_check):
                if re.search(r'\b\d{1,3}%\b', line_content_to_check) or \
                   re.search(r'\b(?:[KMGT]i?B/s|bytes/s)\b', line_content_to_check) or \
                   re.search(r'\([\d.]+[KMGT]?B\)', line_content_to_check):
                    return True
            
            if any(kw in line_content_to_check for kw in pip_keywords):
                 if not ("-I-" in line_content_to_check or "-D-" in line_content_to_check or \
                         any(err_kw in line_content_to_check.lower() for err_kw in ["error", "failed", "warning"])):
                    if re.search(r'\([\d.]+\s*[KMGT]?B\)', line_content_to_check): # e.g. Collecting foo (10kB)
                        return True
        return False

    def _get_log_type_hint(self, message: str, warrior_context: dict) -> str:
        if warrior_context.get("warrior_context_id"):
            if self._project_pattern.search(message): return "WARRIOR_PROJECT_STATUS"
            if self._testsuite_pattern.search(message): return "WARRIOR_TESTSUITE_STATUS"
            if self._testcase_pattern.search(message): return "WARRIOR_TESTCASE_STATUS"
            if self._keyword_status_pattern.search(message): return "WARRIOR_KEYWORD_STATUS"
            if self._command_status_pattern.search(message): return "WARRIOR_CMD_STATUS"
            if self._keyword_start_pattern.match(message): return "WARRIOR_KEYWORD_START"
            if self._testsuite_start_pattern.match(message): return "WARRIOR_TESTSUITE_START"
            if self._testcase_start_pattern.match(message): return "WARRIOR_TESTCASE_START"
            if message.startswith("-I- Command #"): return "WARRIOR_CMD_DETAILS"
            if message.startswith("syntax error:"): return "WARRIOR_SYNTAX_ERROR"
            if "[error]" in message and "Endprompt" not in message : return "WARRIOR_ERROR_TAG"
            if message.startswith("-I-"): return "WARRIOR_INFO"
            if message.startswith("-D-"): return "WARRIOR_DEBUG"
            if message.startswith("-W-"): return "WARRIOR_WARN"
            if message.startswith("-E-"): return "WARRIOR_ERROR_PREFIX"
            if message.startswith("-N-"): return "WARRIOR_NOTICE"
            if message.strip() in ["#", "##", "# [edit]", "# fujitsu@DCN-fujitsu% '"] or not message.strip(): 
                return "WARRIOR_SHELL_PROMPT_OR_EMPTY"
            if message.startswith("----") or message.startswith("====") or message.startswith("++++"): 
                return "WARRIOR_SEPARATOR"
            return "WARRIOR_GENERIC"

        if message.startswith("[Pipeline]"): return "JENKINS_PIPELINE"
        if message.startswith("+ "): return "JENKINS_SHELL_CMD"
        
        # Check for git_like_progress specifically if it's not a warrior context
        # This helps assign a more specific hint if the line wasn't filtered but is git progress
        if self._git_like_progress_pattern.match(message): return "GIT_PROGRESS_OUTPUT"

        if message.startswith(" > git"):
            if "# 'git version" in message: return "JENKINS_GIT_OUTPUT"
            return "JENKINS_GIT_CMD"
        if any(s in message for s in ["Cloning into", "Note: checking out", "You are in 'detached HEAD' state"]):
            return "JENKINS_GIT_OUTPUT"
        if message.lower().startswith("warning:"): return "JENKINS_WARNING"
        if any(s in message for s in ["Collecting pip", "Downloading", "Installing collected packages", "Successfully installed"]):
            return "PIP_INSTALL_INFO" 
        
        if re.match(r"^\[\s*\d+\.\d+\]", message): return "SYSTEM_KERNEL_MSG"
        if re.match(r"^\w+\[\d+\]:", message): return "SYSTEM_PROCESS_MSG"
        return "GENERIC_LOG_MESSAGE"

    def _update_hierarchical_context(self, message_content: str, current_context: dict):
        proj_match = self._project_pattern.search(message_content)
        if proj_match:
            current_context["project_id"] = proj_match.group(1).strip()
            current_context["project_status"] = proj_match.group(2).strip()
            current_context.update({
                "test_suite_id": None, "test_suite_status": None,
                "test_case_id": None, "test_case_status": None,
                "keyword_name": None, "step_id_from_msg": None,
                "step_description": None, "step_status": None })
            return
        ts_start_match = self._testsuite_start_pattern.search(message_content)
        if ts_start_match:
            current_context["test_suite_id"] = ts_start_match.group(1).strip()
            current_context["test_suite_status"] = None 
            current_context.update({
                "test_case_id": None, "test_case_status": None,
                "keyword_name": None, "step_id_from_msg": None,
                "step_description": None, "step_status": None })
            return
        ts_match = self._testsuite_pattern.search(message_content)
        if ts_match:
            current_context["test_suite_id"] = ts_match.group(1).strip()
            current_context["test_suite_status"] = ts_match.group(2).strip()
            current_context.update({
                "test_case_id": None, "test_case_status": None,
                "keyword_name": None, "step_id_from_msg": None,
                "step_description": None, "step_status": None })
            return
        tc_start_match = self._testcase_start_pattern.search(message_content)
        if tc_start_match:
            current_context["test_case_id"] = tc_start_match.group(1).strip()
            current_context["test_case_status"] = None
            current_context.update({
                "keyword_name": None, "step_id_from_msg": None,
                "step_description": None, "step_status": None })
            return
        tc_match = self._testcase_pattern.search(message_content)
        if tc_match:
            current_context["test_case_id"] = tc_match.group(1).strip()
            current_context["test_case_status"] = tc_match.group(2).strip()
            current_context.update({
                "keyword_name": None, "step_id_from_msg": None,
                "step_description": None, "step_status": None })
            return
        kw_start_match = self._keyword_start_pattern.search(message_content)
        if kw_start_match:
            current_context["keyword_name"] = kw_start_match.group(1).strip()
            current_context["step_status"] = None 
            return
        kw_status_match = self._keyword_status_pattern.search(message_content)
        if kw_status_match:
            current_context["step_status"] = kw_status_match.group(2).strip()
            return
        cmd_status_match = self._command_status_pattern.search(message_content)
        if cmd_status_match and not current_context.get("step_status"): 
            current_context["step_status"] = cmd_status_match.group(1).strip()
            return
        step_num_match = self._step_number_pattern.search(message_content)
        if step_num_match:
            current_context["step_id_from_msg"] = step_num_match.group(1).strip()
            return
        step_desc_match = self._step_desc_pattern.search(message_content)
        if step_desc_match:
            current_context["step_description"] = step_desc_match.group(1).strip()
            return

    def _process_single_log_file(self, log_file_path: Path):
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = self.output_base_dir / f"{log_file_path.stem}.jsonl"
        logger.info(f"Processing file: {log_file_path} -> {output_file_path}")
        
        processed_events_count = 0
        current_event_buffer: list[tuple[str, datetime|None, str|None, dict]] = [] 
        current_group_raw_timestamp_str = None 
        
        current_hierarchical_context = {
            "project_id": None, "project_status": None,
            "test_suite_id": None, "test_suite_status": None,
            "test_case_id": None, "test_case_status": None,
            "keyword_name": None, 
            "step_id_from_msg": None, 
            "step_description": None,
            "step_status": None 
        }
        try:
            try: total_lines = sum(1 for _ in open(log_file_path, 'r', encoding='utf-8', errors='replace'))
            except Exception: total_lines = None 

            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as infile, \
                 open(output_file_path, 'w', encoding='utf-8') as outfile:
                line_iterator = tqdm(enumerate(infile), total=total_lines, desc=f"Lines in {log_file_path.name}", unit="line", leave=False)
                
                for line_number, raw_line in line_iterator:
                    line = raw_line.rstrip('\n\r')
                    if not line.strip() and not current_event_buffer: continue
                    
                    dt_obj_curr, msg_curr, raw_ts_curr, warrior_ctx_curr = self._extract_dt_and_message_from_line(line)
                    
                    if self._is_line_to_be_filtered(msg_curr, warrior_ctx_curr):
                        is_new_group_due_to_ts_change_for_skipped_line = False
                        if current_event_buffer:
                            if raw_ts_curr is not None and current_group_raw_timestamp_str is not None and \
                               raw_ts_curr != current_group_raw_timestamp_str:
                                is_new_group_due_to_ts_change_for_skipped_line = True
                            elif raw_ts_curr is not None and current_group_raw_timestamp_str is None:
                                is_new_group_due_to_ts_change_for_skipped_line = True 
                        
                        if is_new_group_due_to_ts_change_for_skipped_line and current_event_buffer:
                             self._write_grouped_event(outfile, current_event_buffer, current_hierarchical_context)
                             processed_events_count +=1
                             current_event_buffer = []
                        if is_new_group_due_to_ts_change_for_skipped_line: # Reset for next group if current one was flushed
                            current_group_raw_timestamp_str = None
                        continue 

                    is_new_group = False
                    if not current_event_buffer: is_new_group = True
                    elif raw_ts_curr is not None and current_group_raw_timestamp_str is not None and \
                            raw_ts_curr != current_group_raw_timestamp_str: is_new_group = True
                    elif raw_ts_curr is not None and current_group_raw_timestamp_str is None: is_new_group = True 
                    
                    if is_new_group and current_event_buffer:
                        self._write_grouped_event(outfile, current_event_buffer, current_hierarchical_context)
                        processed_events_count +=1
                        current_event_buffer = []
                    
                    self._update_hierarchical_context(msg_curr, current_hierarchical_context)
                    current_event_buffer.append((msg_curr, dt_obj_curr, raw_ts_curr, warrior_ctx_curr))
                    
                    if is_new_group or (not current_group_raw_timestamp_str and raw_ts_curr):
                        current_group_raw_timestamp_str = raw_ts_curr
                
                if current_event_buffer:
                    self._write_grouped_event(outfile, current_event_buffer, current_hierarchical_context)
                    processed_events_count +=1
            
            logger.info(f"Finished processing {log_file_path}. {processed_events_count} events written to {output_file_path}")
        except Exception as e:
            logger.error(f"Error processing file {log_file_path}: {e}", exc_info=True)

    def _write_grouped_event(self, outfile, 
                             event_buffer: list[tuple[str, datetime|None, str|None, dict]],
                             active_hier_context: dict):
        if not event_buffer: return

        group_timestamp_obj = None
        group_warrior_context_from_buffer = {} 
        
        if event_buffer[0][1]: group_timestamp_obj = event_buffer[0][1]
        else:
            for _, dt_obj, _, _ in event_buffer:
                if dt_obj:
                    group_timestamp_obj = dt_obj
                    break
        if not group_timestamp_obj:
             group_timestamp_obj = datetime.now(timezone.utc) 

        for _, _, _, w_ctx_line in event_buffer:
            if w_ctx_line.get('warrior_context_id'):
                group_warrior_context_from_buffer = w_ctx_line
                break 
        
        message_parts = [msg_content for msg_content, _, _, _ in event_buffer if msg_content]
        if not message_parts: return

        combined_message_content = " <NL> ".join(message_parts)
        if not combined_message_content.strip(): return

        normalized_ts_str = self._normalize_timestamp_to_utc_iso(group_timestamp_obj)
        log_type = self._get_log_type_hint(combined_message_content, group_warrior_context_from_buffer) 

        event_specific_hier_info = dict(active_hier_context) 

        if group_warrior_context_from_buffer.get('warrior_context_id'):
            event_specific_hier_info.setdefault("warrior_context_tag", group_warrior_context_from_buffer.get('warrior_context_id'))
            event_specific_hier_info.setdefault("warrior_node_id", group_warrior_context_from_buffer.get('warrior_node_id'))
            if event_specific_hier_info.get("step_id_from_msg"):
                 event_specific_hier_info["step_id"] = event_specific_hier_info.pop("step_id_from_msg")
            elif not event_specific_hier_info.get("step_id") and group_warrior_context_from_buffer.get('warrior_step_id'):
                 event_specific_hier_info["step_id"] = group_warrior_context_from_buffer.get('warrior_step_id')
        elif event_specific_hier_info.get("step_id_from_msg"):
            event_specific_hier_info["step_id"] = event_specific_hier_info.pop("step_id_from_msg")

        processed_event = {
            "timestamp_utc": normalized_ts_str,
            "log_type_hint": log_type,
            **{k: v for k, v in event_specific_hier_info.items() if v is not None and k != "step_id_from_msg"}, 
            "message_content": combined_message_content
        }
        if "step_id_from_msg" in processed_event:
            del processed_event["step_id_from_msg"]

        outfile.write(json.dumps(processed_event) + "\n")

    def process_logs(self, num_files_to_process: int = 0): 
        files_in_index = self.manage_index()
        if not files_in_index:
            logger.info("No log files to process.")
            return

        actual_files_to_process = files_in_index
        if num_files_to_process > 0 and num_files_to_process < len(files_in_index):
            actual_files_to_process = files_in_index[:num_files_to_process]
        
        if not actual_files_to_process:
            logger.info("Effective number of files to process is 0.")
            return

        logger.info(f"Starting preprocessing for {len(actual_files_to_process)} log file(s)...")
        for log_file_path in tqdm(actual_files_to_process, desc="Processing Files", unit="file"):
            if not log_file_path.exists():
                logger.warning(f"File listed in index does not exist, skipping: {log_file_path}")
                continue
            self._process_single_log_file(log_file_path)
        logger.info("Log preprocessing complete.")



def main():
    current_dir = Path.cwd()
    sample_data_dir = current_dir / "data" / "dcn" 

    preprocessor = LogPreprocessor(data_directory=sample_data_dir.relative_to(current_dir))
    preprocessor.use_index_cache = False 
    # To see filtering decisions for debugging:
    # logging.getLogger().setLevel(logging.DEBUG)
    preprocessor.process_logs() 


if __name__ == '__main__':
    main()