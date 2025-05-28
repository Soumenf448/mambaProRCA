import re
import json
from pathlib import Path
from datetime import datetime, timezone
import csv
import logging
from tqdm import tqdm

try:
    from dateutil import parser as dateutil_parser
except ImportError:
    dateutil_parser = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("LogPreprocessor")
logger.setLevel(logging.INFO)

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
            r'(\d{8}\.\d{6})\s+(stdout|stderr):([a-zA-Z0-9\._-]+?)(?:\.Warrior)?\s*#\s*(.*)'
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
        self._git_like_progress_pattern = re.compile(
            r"^\s*(?:remote:\s*)?"
            r"(?:Counting|Compressing|Receiving|Resolving)\s+(?:objects|deltas):\s+"
            r"\d{1,3}%\s*\(\d+/\d+\)"
            r"(?:,\s*\S.*?)*?"
            r"\s*(?:\x1b\[K)?\s*$"
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
        self._project_start_pattern = re.compile(r"^-I- \${5,}\s*START OF PROJECT\s*:\s*(\S+)\s*\${5,}")
        self._testsuite_pattern = re.compile(r"^-I- Testsuite:(\S+)\s+STATUS:(\w+)")
        self._testcase_pattern = re.compile(r"^-I- TESTCASE:(\S+)\s+STATUS:(\w+)")
        self._keyword_status_pattern = re.compile(r"^-I- KEYWORD:(\S+)\s+STATUS:(\w+)")
        self._command_status_pattern = re.compile(r"^-I- COMMAND STATUS:(\w+)")
        self._step_number_pattern = re.compile(r"^-I- step number:\s*(\d+)")
        self._step_desc_pattern = re.compile(r"^-I- Teststep Description:\s*(.+)")
        self._keyword_start_pattern = re.compile(r"^-I- \*{5,}\s*Keyword:\s*(\S+)\s*\*{5,}")
        self._testsuite_start_pattern = re.compile(r"^-I- \${5,}\s*START OF TEST SUITE\s*:\s*(\S+)\s*\${5,}")
        self._testcase_start_pattern = re.compile(r"^-I- ={5,}\s*START OF TESTCASE\s*:\s*(\S+)\s*={5,}")

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
            self.index_file_path.parent.mkdir(parents=True, exist_ok=True)
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
            
            if dateutil_parser: # Use dateutil if available
                return dateutil_parser.parse(ts_str)
            else: # Fallback for specific formats if dateutil is not installed
                if matched_pattern.pattern == r'\[?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[,\.]\d{1,9}(?:Z|[+-]\d{2}:\d{2})?)\]?':
                    # Simplified ISO 8601 parsing without full timezone offset handling if dateutil is missing
                    # This will be naive if timezone is not Z
                    ts_str_cleaned = ts_str.strip('[]')
                    if 'Z' in ts_str_cleaned:
                        ts_str_cleaned = ts_str_cleaned.replace('Z', '')
                    elif '+' in ts_str_cleaned or '-' in ts_str_cleaned[10:]: # Check for timezone offset beyond date
                        # Basic handling: try to parse up to seconds or milliseconds, ignore offset for naive datetime
                        ts_part = ts_str_cleaned.split_datetime = ts_str_cleaned.split('+')[0].split('-')[0] if '+' in ts_str_cleaned else ts_str_cleaned.split('-')[0]
                        ts_str_cleaned = ts_part[0]

                    if '.' in ts_str_cleaned:
                        return datetime.strptime(ts_str_cleaned, '%Y-%m-%dT%H:%M:%S.%f')
                    elif ',' in ts_str_cleaned: # Handle comma as millisecond separator
                         return datetime.strptime(ts_str_cleaned, '%Y-%m-%dT%H:%M:%S,%f')
                    else:
                        return datetime.strptime(ts_str_cleaned, '%Y-%m-%dT%H:%M:%S')
                elif matched_pattern.pattern == r'([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?\s+\d{4})':
                    try:
                        return datetime.strptime(ts_str, '%b %d %H:%M:%S.%f %Y')
                    except ValueError:
                        return datetime.strptime(ts_str, '%b %d %H:%M:%S %Y')
                # Add more strptime fallbacks if needed for other specific formats
                logger.debug(f"dateutil.parser not available and no specific strptime match for: {ts_str} with pattern {matched_pattern.pattern}")
                return None

        except Exception as e:
            logger.debug(f"Could not parse timestamp string '{ts_str}' with pattern '{matched_pattern.pattern}': {e}")
            return None

    def _extract_dt_and_message_from_line(self, line: str) -> tuple[datetime | None, str, str | None, dict]:
        message_part = line
        dt_obj = None
        raw_ts_str_external = None
        warrior_context = {}

        for pattern in self._timestamp_patterns:
            match = pattern.match(line)
            if match:
                raw_ts_str_external = match.group(1)
                dt_obj = self._parse_datetime_from_string(raw_ts_str_external, pattern)
                if dt_obj:
                    message_part = line[match.end():].lstrip()
                    break

        warrior_match = self._warrior_prefix_pattern.match(message_part)
        if warrior_match:
            warrior_ts_internal_str, stream_type, context_id_str, actual_message = warrior_match.groups()
            warrior_context['warrior_timestamp_internal'] = warrior_ts_internal_str
            warrior_context['warrior_stream'] = stream_type
            warrior_context['warrior_context_id'] = context_id_str.strip()
            
            parts = context_id_str.strip().split('.')
            if len(parts) > 0: warrior_context['warrior_node_id'] = parts[0]
            if len(parts) > 1: warrior_context['warrior_project_id_from_prefix'] = parts[1]
            if len(parts) > 2: warrior_context['warrior_suite_id_from_prefix'] = parts[2]
            if len(parts) > 3: warrior_context['warrior_case_id_from_prefix'] = parts[3]
            if len(parts) > 4: warrior_context['warrior_keyword_or_step_from_prefix'] = ".".join(parts[4:])
            
            message_part = actual_message

            if not dt_obj and self._timestamp_patterns[-1].match(warrior_ts_internal_str):
                 dt_obj_internal = self._parse_datetime_from_string(warrior_ts_internal_str, self._timestamp_patterns[-1])
                 if dt_obj_internal:
                     dt_obj = dt_obj_internal
        
        return dt_obj, message_part.strip(), raw_ts_str_external, warrior_context

    def _normalize_timestamp_to_utc_iso(self, dt_obj: datetime | None) -> str | None:
        if not dt_obj: return None
        if dt_obj.tzinfo is None: dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        else: dt_obj = dt_obj.astimezone(timezone.utc)
        return dt_obj.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _is_line_to_be_filtered(self, line_content_to_check: str, warrior_context: dict) -> bool:
        # (Same as previous full version)
        if self._ansi_cursor_pattern.match(line_content_to_check): return True
        if line_content_to_check == "\x1b[K" or line_content_to_check == "\x1b[0m": return True
        if self._detailed_progress_bar_pattern.match(line_content_to_check): return True
        if self._git_like_progress_pattern.match(line_content_to_check): return True
        bar_char_matches = self._progress_bar_chars_pattern.findall(line_content_to_check)
        if len(bar_char_matches) >= 2 and self._strong_progress_indicators.search(line_content_to_check): return True
        if self._percentage_progress_pattern.search(line_content_to_check): return True
        if self._standalone_percentage_pattern.match(line_content_to_check): return True
        if self._curl_progress_header_pattern.match(line_content_to_check): return True
        if self._curl_progress_stats_pattern.match(line_content_to_check): return True
        pip_keywords = ["Collecting ", "Downloading https://", "Installing collected packages:", "Successfully installed", "Found existing installation:", "Uninstalling", "Successfully uninstalled", "You are using pip version", "You should consider upgrading via the", "Looking in indexes:"]
        if not warrior_context.get('warrior_context_id'): # Only apply pip filtering if not a warrior message
            if "Downloading https://" in line_content_to_check and any(ext in line_content_to_check for ext in [".whl", ".tar.gz", ".zip"]):
                if re.search(r'\b\d{1,3}%\b', line_content_to_check) or re.search(r'\b(?:[KMGT]i?B/s|bytes/s)\b', line_content_to_check) or re.search(r'\([\d.]+[KMGT]?B\)', line_content_to_check):
                    return True
            if any(kw in line_content_to_check for kw in pip_keywords):
                 if not ("-I-" in line_content_to_check or "-D-" in line_content_to_check or \
                         any(err_kw in line_content_to_check.lower() for err_kw in ["error", "failed", "warning"])): # Don't filter if it's a warrior log or an error
                    if re.search(r'\([\d.]+\s*[KMGT]?B\)', line_content_to_check): # e.g. Collecting foo (10kB)
                        return True
        return False

    def _get_log_type_hint(self, message: str, warrior_context: dict) -> str:
        # (Same as previous full version)
        if warrior_context.get("warrior_context_id"):
            if self._project_start_pattern.search(message) or self._project_pattern.search(message): return "WARRIOR_PROJECT_CONTEXT"
            if self._testsuite_start_pattern.search(message) or self._testsuite_pattern.search(message): return "WARRIOR_TESTSUITE_CONTEXT"
            if self._testcase_start_pattern.search(message) or self._testcase_pattern.search(message): return "WARRIOR_TESTCASE_CONTEXT"
            if self._keyword_start_pattern.search(message) or self._keyword_status_pattern.search(message): return "WARRIOR_KEYWORD_CONTEXT"
            if self._command_status_pattern.search(message): return "WARRIOR_CMD_STATUS"
            if self._step_number_pattern.search(message) or self._step_desc_pattern.search(message): return "WARRIOR_STEP_INFO"
            if message.startswith("-I- Command #"): return "WARRIOR_CMD_DETAILS"
            if message.startswith("syntax error:"): return "WARRIOR_SYNTAX_ERROR"
            if "[error]" in message and "Endprompt" not in message : return "WARRIOR_ERROR_TAG"
            if message.startswith("-I-"): return "WARRIOR_INFO"
            if message.startswith("-D-"): return "WARRIOR_DEBUG"
            if message.startswith("-W-"): return "WARRIOR_WARN"
            if message.startswith("-E-"): return "WARRIOR_ERROR_PREFIX"
            if message.startswith("-N-"): return "WARRIOR_NOTICE"
            if message.strip() in ["#", "##", "# [edit]", "# fujitsu@DCN-fujitsu% '"] or not message.strip(): return "WARRIOR_SHELL_PROMPT_OR_EMPTY"
            if message.startswith("----") or message.startswith("====") or message.startswith("++++"): return "WARRIOR_SEPARATOR"
            return "WARRIOR_GENERIC"
        if message.startswith("[Pipeline]"): return "JENKINS_PIPELINE"
        if message.startswith("+ "): return "JENKINS_SHELL_CMD"
        if self._git_like_progress_pattern.match(message): return "GIT_PROGRESS_OUTPUT"
        if message.startswith(" > git"):
            if "# 'git version" in message: return "JENKINS_GIT_OUTPUT"
            return "JENKINS_GIT_CMD"
        if any(s in message for s in ["Cloning into", "Note: checking out", "You are in 'detached HEAD' state"]): return "JENKINS_GIT_OUTPUT"
        if message.lower().startswith("warning:"): return "JENKINS_WARNING"
        if any(s in message for s in ["Collecting pip", "Downloading", "Installing collected packages", "Successfully installed"]): return "PIP_INSTALL_INFO"
        if re.match(r"^\[\s*\d+\.\d+\]", message): return "SYSTEM_KERNEL_MSG"
        if re.match(r"^\w+\[\d+\]:", message): return "SYSTEM_PROCESS_MSG"
        return "GENERIC_LOG_MESSAGE"


    def _update_hierarchical_context(self, message_content: str, current_context: dict):
        # (Same as previous version with project_start_pattern)
        proj_start_match = self._project_start_pattern.search(message_content)
        if proj_start_match:
            current_context["project_id"] = proj_start_match.group(1).strip()
            current_context["project_status"] = "RUNNING" 
            current_context.update({
                "test_suite_id": None, "test_suite_status": None,
                "test_case_id": None, "test_case_status": None,
                "keyword_name": None, "step_id_from_msg": None,
                "step_description": None, "step_status": None })
            return
        proj_match = self._project_pattern.search(message_content) 
        if proj_match:
            if current_context.get("project_id") is None or current_context.get("project_id") == proj_match.group(1).strip():
                current_context["project_id"] = proj_match.group(1).strip()
                current_context["project_status"] = proj_match.group(2).strip()
            if current_context.get("project_status", "").upper() != "RUNNING":
                current_context.update({
                    "test_suite_id": None, "test_suite_status": None,
                    "test_case_id": None, "test_case_status": None,
                    "keyword_name": None, "step_id_from_msg": None,
                    "step_description": None, "step_status": None })
            return

        ts_start_match = self._testsuite_start_pattern.search(message_content)
        if ts_start_match:
            current_context["test_suite_id"] = ts_start_match.group(1).strip()
            current_context["test_suite_status"] = "RUNNING"
            current_context.update({
                "test_case_id": None, "test_case_status": None,
                "keyword_name": None, "step_id_from_msg": None,
                "step_description": None, "step_status": None })
            return
        ts_match = self._testsuite_pattern.search(message_content)
        if ts_match:
            if current_context.get("test_suite_id") is None or current_context.get("test_suite_id") == ts_match.group(1).strip():
                current_context["test_suite_id"] = ts_match.group(1).strip()
                current_context["test_suite_status"] = ts_match.group(2).strip()
            if current_context.get("test_suite_status", "").upper() != "RUNNING":
                current_context.update({
                    "test_case_id": None, "test_case_status": None,
                    "keyword_name": None, "step_id_from_msg": None,
                    "step_description": None, "step_status": None })
            return

        tc_start_match = self._testcase_start_pattern.search(message_content)
        if tc_start_match:
            current_context["test_case_id"] = tc_start_match.group(1).strip()
            current_context["test_case_status"] = "RUNNING"
            current_context.update({
                "keyword_name": None, "step_id_from_msg": None,
                "step_description": None, "step_status": None })
            return
        tc_match = self._testcase_pattern.search(message_content)
        if tc_match:
            if current_context.get("test_case_id") is None or current_context.get("test_case_id") == tc_match.group(1).strip():
                current_context["test_case_id"] = tc_match.group(1).strip()
                current_context["test_case_status"] = tc_match.group(2).strip()
            if current_context.get("test_case_status", "").upper() != "RUNNING":
                current_context.update({
                    "keyword_name": None, "step_id_from_msg": None,
                    "step_description": None, "step_status": None })
            return

        kw_start_match = self._keyword_start_pattern.search(message_content)
        if kw_start_match:
            current_context["keyword_name"] = kw_start_match.group(1).strip()
            current_context["step_status"] = "RUNNING" 
            current_context.update({ "step_id_from_msg": None, "step_description": None })
            return
        kw_status_match = self._keyword_status_pattern.search(message_content)
        if kw_status_match:
            if current_context.get("keyword_name") is None or current_context.get("keyword_name") == kw_status_match.group(1).strip():
                current_context["keyword_name"] = kw_status_match.group(1).strip()
                current_context["step_status"] = kw_status_match.group(2).strip()
            if current_context.get("step_status", "").upper() != "RUNNING":
                current_context.update({ "keyword_name": None, "step_id_from_msg": None, "step_description": None})
            return
        
        cmd_status_match = self._command_status_pattern.search(message_content)
        if cmd_status_match:
            current_context["step_status"] = cmd_status_match.group(1).strip()
            # Don't return immediately, could be other step info on the same line/event

        step_num_match = self._step_number_pattern.search(message_content)
        if step_num_match:
            current_context["step_id_from_msg"] = step_num_match.group(1).strip()
            current_context["step_description"] = None 
            current_context["step_status"] = "RUNNING" # Assume step starts running
            return 

        step_desc_match = self._step_desc_pattern.search(message_content)
        if step_desc_match:
            current_context["step_description"] = step_desc_match.group(1).strip()
            # Don't necessarily return as status might follow

    def _write_grouped_event(self, outfile,
                             event_buffer: list[tuple[str, datetime | None, str | None, dict]],
                             active_hier_context: dict):
        if not event_buffer: return

        group_timestamp_obj = None
        group_warrior_context_from_buffer = {} 

        if event_buffer[0][1]: group_timestamp_obj = event_buffer[0][1]
        else:
            for _, dt_obj, _, _ in event_buffer:
                if dt_obj:
                    group_timestamp_obj = dt_obj; break
        if not group_timestamp_obj: group_timestamp_obj = datetime.now(timezone.utc)
        normalized_ts_str = self._normalize_timestamp_to_utc_iso(group_timestamp_obj)

        for _, _, _, w_ctx_line in event_buffer:
            if w_ctx_line.get('warrior_context_id'):
                group_warrior_context_from_buffer = w_ctx_line; break
        
        message_parts = [msg_content for msg_content, _, _, _ in event_buffer if msg_content]
        if not message_parts: return
        combined_message_content = " <NL> ".join(message_parts)
        if not combined_message_content.strip(): return
        
        log_type = self._get_log_type_hint(combined_message_content, group_warrior_context_from_buffer)

        final_event_dict = {"timestamp_utc": normalized_ts_str, "log_type_hint": log_type}

        # Populate from active_hier_context (from -I- tags) first
        for key in ["project_id", "project_status", "test_suite_id", "test_suite_status",
                    "test_case_id", "test_case_status", "keyword_name", 
                    "step_id_from_msg", "step_description", "step_status"]:
            if active_hier_context.get(key) is not None:
                # Rename step_id_from_msg to step_id for the output
                if key == "step_id_from_msg":
                    final_event_dict["step_id"] = active_hier_context.get(key)
                else:
                    final_event_dict[key] = active_hier_context.get(key)

        # Augment/Fallback with warrior prefix context
        if group_warrior_context_from_buffer:
            final_event_dict.setdefault("warrior_context_tag", group_warrior_context_from_buffer.get('warrior_context_id'))
            final_event_dict.setdefault("warrior_node_id", group_warrior_context_from_buffer.get('warrior_node_id'))
            
            final_event_dict.setdefault("project_id", group_warrior_context_from_buffer.get('warrior_project_id_from_prefix'))
            final_event_dict.setdefault("test_suite_id", group_warrior_context_from_buffer.get('warrior_suite_id_from_prefix'))
            final_event_dict.setdefault("test_case_id", group_warrior_context_from_buffer.get('warrior_case_id_from_prefix'))
            
            # Fallback for keyword or step_id
            kw_or_step_prefix = group_warrior_context_from_buffer.get('warrior_keyword_or_step_from_prefix')
            if kw_or_step_prefix:
                # Heuristic: if it contains alpha, more likely a keyword if keyword_name is not set
                if final_event_dict.get("keyword_name") is None and any(c.isalpha() for c in kw_or_step_prefix):
                    final_event_dict["keyword_name"] = kw_or_step_prefix
                # If step_id is still not set by specific tags or a more specific prefix part
                if final_event_dict.get("step_id") is None:
                    final_event_dict["step_id"] = kw_or_step_prefix
            elif final_event_dict.get("step_id") is None and group_warrior_context_from_buffer.get('warrior_step_id'): # original broader step_id
                 final_event_dict["step_id"] = group_warrior_context_from_buffer.get('warrior_step_id')
        
        # Remove any keys that ended up with None value after all attempts
        keys_to_remove = [k for k, v in final_event_dict.items() if v is None]
        for k in keys_to_remove:
            del final_event_dict[k]
            
        final_event_dict["message_content"] = combined_message_content
        outfile.write(json.dumps(final_event_dict) + "\n")

    def _process_single_log_file(self, log_file_path: Path):
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = log_file_path.parent.parent / f"{log_file_path.parent.stem}_jsonl" / f"{log_file_path.stem}.jsonl"
        logger.info(f"Processing file: {log_file_path} -> {output_file_path}")

        processed_events_count = 0
        current_event_buffer: list[tuple[str, datetime | None, str | None, dict]] = []
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
            try:
                total_lines = sum(1 for _ in open(log_file_path, 'r', encoding='utf-8', errors='replace'))
            except Exception:
                total_lines = None

            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as infile, \
                 open(output_file_path, 'w', encoding='utf-8') as outfile:
                line_iterator = tqdm(enumerate(infile), total=total_lines, desc=f"Lines in {log_file_path.name}", unit="line", leave=False)

                for line_number, raw_line in line_iterator:
                    line = raw_line.rstrip('\n\r')
                    if not line.strip() and not current_event_buffer: continue

                    dt_obj_curr, msg_curr, raw_ts_curr, warrior_ctx_curr = self._extract_dt_and_message_from_line(line)
                    
                    # Update hierarchical context BEFORE filtering or grouping, so current line's tags affect context
                    self._update_hierarchical_context(msg_curr, current_hierarchical_context)

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
                        if is_new_group_due_to_ts_change_for_skipped_line:
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
                    
                    current_event_buffer.append((msg_curr, dt_obj_curr, raw_ts_curr, warrior_ctx_curr))
                    
                    if is_new_group or (not current_group_raw_timestamp_str and raw_ts_curr):
                        current_group_raw_timestamp_str = raw_ts_curr
                
                if current_event_buffer:
                    self._write_grouped_event(outfile, current_event_buffer, current_hierarchical_context)
                    processed_events_count +=1
            
            logger.info(f"Finished processing {log_file_path}. {processed_events_count} events written to {output_file_path}")
        except Exception as e:
            logger.error(f"Error processing file {log_file_path}: {e}", exc_info=True)

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


if __name__ == '__main__':
    current_dir = Path.cwd()
    sample_data_dir = current_dir / "data/dcn" 
  
    # Initialize preprocessor with the directory containing the sample log file
    preprocessor = LogPreprocessor(data_directory=sample_data_dir) # Use the specific test dir
    preprocessor.use_index_cache = False # Force rebuild of index for the test
    preprocessor.process_logs()
