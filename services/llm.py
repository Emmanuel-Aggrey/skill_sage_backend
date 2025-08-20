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
                except Exception:
                    self.call_counts[i] = 0

    def _log_api_call(self, key_index, success=True):
        """Log API call to the respective key's file"""
        self.call_counts[key_index] += 1
        log_file = f"logs/api_key_{key_index+1}_calls.txt"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if success else "FAILED"

        try:
            with open(log_file, 'a') as f:
                f.write(
                    f"{timestamp} - Call #{self.call_counts[key_index]} - {status}\n")
                f.write(f"Total calls: {self.call_counts[key_index]}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")

    def _get_next_available_key(self):
        """Find the next available key that hasn't failed"""
        available_keys = [i for i in range(
            len(self.api_keys)) if i not in self.failed_keys]

        if not available_keys:
            return None

        # Find next key after current, or first available if current is last
        current_index = None
        for i, key_idx in enumerate(available_keys):
            if key_idx == self.current_key:
                current_index = i
                break

        if current_index is not None and current_index + 1 < len(available_keys):
            return available_keys[current_index + 1]
        else:
            # Return first available key (cycling back)
            return available_keys[0] if available_keys[0] != self.current_key else None

    def _switch_to_next_key(self):
        """Switch to next available key"""
        next_key = self._get_next_available_key()
        if next_key is not None:
            self.current_key = next_key
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
            print(message)
            sendError(message)
            raise Exception(message)

        # Allow more attempts for retries
        max_attempts = len(self.api_keys) * 2

        self.request_count = 0

        for attempt in range(max_attempts):
            try:
                # Skip failed keys
                if self.current_key in self.failed_keys:
                    if not self._switch_to_next_key():
                        break
                    continue

                self.request_count += 1
                print(
                    f"🔄 Attempting with key #{self.current_key + 1} (attempt {self.request_count + 1})")

                # Make request
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                headers = {
                    'Content-Type': 'application/json',
                    'X-goog-api-key': self.api_keys[self.current_key]
                }

                self.last_request = time.time()
                response = requests.post(self.api_url, headers=headers,
                                         data=json.dumps(payload), timeout=120)

                # Handle successful response
                if response.status_code == 200:
                    result = response.json()
                    print(
                        f"✅ Success with key #{self.current_key + 1} on attempt {self.request_count + 1}")

                    # Log successful call
                    self._log_api_call(self.current_key, True)

                    return result['candidates'][0]['content']['parts'][0]['text'].strip()

                # Handle quota exceeded
                elif response.status_code == 429:
                    message = f"🚫 Key #{self.current_key + 1} quota exceeded"
                    print(message)

                    # Log failed call and mark key as failed
                    self._log_api_call(self.current_key, False)
                    self.failed_keys.add(self.current_key)

                    # Try to switch to next key
                    if self._switch_to_next_key():
                        continue
                    else:
                        # No more keys available
                        break

                # Handle other HTTP errors
                else:
                    self._log_api_call(self.current_key, False)
                    error_message = f"HTTP {response.status_code}: {response.text[:100]}"
                    print(
                        f"❗ Key #{self.current_key + 1} error: {error_message}")

                    # For non-quota errors, try next key but don't mark current as permanently failed
                    if self._switch_to_next_key():
                        time.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        raise Exception(f"Request failed: {error_message}")

            except requests.RequestException as e:
                # Network/request errors
                self._log_api_call(self.current_key, False)
                error_message = f"Request error with key #{self.current_key + 1}: {str(e)[:100]}"
                print(error_message)

                # Try next key for network errors
                if self._switch_to_next_key():
                    time.sleep(2)  # Longer pause for network issues
                    continue
                else:
                    raise Exception(error_message)

            except Exception as e:
                # Other unexpected errors
                self._log_api_call(self.current_key, False)
                error_message = f"Unexpected error with key #{self.current_key + 1}: {str(e)[:100]}"
                print(error_message)

                # Try next key
                if attempt < max_attempts - 1 and self._switch_to_next_key():
                    time.sleep(1)
                    continue
                else:
                    raise Exception(error_message)

        # If we get here, all attempts failed
        message = f"💥 Failed after {max_attempts} attempts across all available keys"
        print(message)
        sendError(message)
        raise Exception(message)

    def stats(self):
        """Show current stats including call counts"""
        active = len(self.api_keys) - len(self.failed_keys)
        failed_keys_list = [f"#{i+1}" for i in self.failed_keys]
        call_info = ", ".join(
            [f"Key #{i+1}: {self.call_counts[i]} calls" for i in range(len(self.api_keys))])

        status = f"📊 Keys: {active}/{len(self.api_keys)} active, current: #{self.current_key + 1}"
        if failed_keys_list:
            status += f", failed: {', '.join(failed_keys_list)}"
        status += f" | {call_info}"

        return status

    def reset_failed_keys(self):
        """Reset failed keys (useful for testing or if quotas reset)"""
        self.failed_keys.clear()
        print("🔄 Reset all failed keys")

    def get_available_keys(self):
        """Get list of available (non-failed) keys"""
        return [i for i in range(len(self.api_keys)) if i not in self.failed_keys]


llm_client = BaseLLMClient()
