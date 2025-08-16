import time
import json
import requests
import os
from datetime import datetime
from decouple import config
from routes.helpers import sendError

# API Keys List
API_KEYS = [
    config("GEMINI_API_KEY_01"),
    config("GEMINI_API_KEY_02"),
]


class BaseLLMClient:
    _instance = None
    _initialized = False

    def __new__(cls, api_keys=None, model="gemini-2.0-flash"):
        if cls._instance is None:
            cls._instance = super(BaseLLMClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_keys=None, model="gemini-2.0-flash"):
        if not self._initialized:
            self.api_keys = api_keys or API_KEYS
            self.model = model
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            self.current_key = 0
            self.failed_keys = set()
            self.last_request = None
            self.call_counts = {}  # Track calls per key
            self._init_log_files()
            BaseLLMClient._initialized = True
            print(f"✅ Loaded {len(self.api_keys)} API keys")

    def _init_log_files(self):
        """Initialize log files for each API key"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        for i in range(len(self.api_keys)):
            log_file = f"logs/api_key_{i+1}_calls.txt"
            self.call_counts[i] = 0

            # Read existing count if file exists
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Get the last line which should contain the total count
                            last_line = lines[-1].strip()
                            if last_line.startswith("Total calls:"):
                                self.call_counts[i] = int(
                                    last_line.split(":")[1].strip())
                except:
                    self.call_counts[i] = 0

    def _log_api_call(self, key_index, success=True):
        """Log API call to the respective key's file"""
        self.call_counts[key_index] += 1
        log_file = f"logs/api_key_{key_index+1}_calls.txt"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if success else "FAILED"

        with open(log_file, 'a') as f:
            f.write(
                f"{timestamp} - Call #{self.call_counts[key_index]} - {status}\n")
            f.write(f"Total calls: {self.call_counts[key_index]}\n")

    def _next_key(self):
        """Switch to next available key"""
        for _ in range(len(self.api_keys)):
            self.current_key = (self.current_key + 1) % len(self.api_keys)
            if self.current_key not in self.failed_keys:
                print(f"🔄 Switched to key #{self.current_key + 1}")
                return True
        return False

    def query_llm(self, prompt):
        """Query LLM with automatic key switching"""

        # Rate limiting
        if self.last_request:
            elapsed = time.time() - self.last_request
            if elapsed < 4:
                time.sleep(4 - elapsed)

        # Check if all keys failed
        if len(self.failed_keys) >= len(self.api_keys):
            message = f"❌ All {len(self.api_keys)} API keys exhausted"
            sendError(message)
            raise Exception(message)

        # Try each key
        for attempt in range(len(self.api_keys) + 2):
            try:
                # Skip failed keys
                if self.current_key in self.failed_keys:
                    if not self._next_key():
                        break

                # Make request
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                headers = {
                    'Content-Type': 'application/json',
                    'X-goog-api-key': self.api_keys[self.current_key]
                }

                self.last_request = time.time()
                response = requests.post(self.api_url, headers=headers,
                                         data=json.dumps(payload), timeout=120)

                # Log the API call attempt
                success = response.status_code == 200
                self._log_api_call(self.current_key, success)

                # Handle response
                if response.status_code == 200:
                    result = response.json()
                    print(
                        f"✅ Success with key #{self.current_key + 1} attempts {attempt + 1}")

                    return result['candidates'][0]['content']['parts'][0]['text'].strip()

                elif response.status_code == 429:
                    message = f"🚫 Key #{self.current_key + 1} quota exceeded"
                    print(message)
                    sendError(message)
                    self.failed_keys.add(self.current_key)
                    if not self._next_key():
                        break
                    continue

                else:
                    response.raise_for_status()

            except Exception as e:
                # Log failed call
                self._log_api_call(self.current_key, False)

                message = f"❗ Error with key #{self.current_key + 1}: {str(e)[:50]}"
                print(message)
                sendError(message)
                if attempt < len(self.api_keys):
                    if not self._next_key():
                        break
                    time.sleep(1)
                    continue
                raise e
        message = f"💥 Failed after trying all {len(self.api_keys)} keys"
        sendError(message)
        raise Exception(message)

    def stats(self):
        """Show current stats including call counts"""
        active = len(self.api_keys) - len(self.failed_keys)
        call_info = ", ".join(
            [f"Key #{i+1}: {self.call_counts[i]} calls" for i in range(len(self.api_keys))])
        return f"📊 Keys: {active}/{len(self.api_keys)} active, current: #{self.current_key + 1} | {call_info}"


llm_client = BaseLLMClient()
