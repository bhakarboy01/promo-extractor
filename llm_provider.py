import os
import re
import time
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

# Automatically load environment variables from current directory and Test folder
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "Test", ".env"))

# Setup dedicated pipeline logger
LOG_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pipeline_run.log")
EXTRAS_DIR = os.path.join(LOG_DIR, "extras")
os.makedirs(EXTRAS_DIR, exist_ok=True)
LLM_CHATS_DIR = EXTRAS_DIR
LLM_CHATS_FILE = os.path.join(EXTRAS_DIR, "llm_chats.json")
if not os.path.exists(LLM_CHATS_FILE):
    alt_chats = os.path.join(os.path.dirname(__file__), "docs", "llm_chats.json")
    if os.path.exists(alt_chats):
        LLM_CHATS_FILE = alt_chats

__all__ = [
    "LOG_FILE",
    "LLM_CHATS_FILE",
    "LOG_DIR",
    "EXTRAS_DIR",
    "log_event",
    "record_llm_interaction",
    "resolve_api_keys",
    "APIKeyPool",
    "BaseLLMProvider",
    "LiteLLMProvider",
]

logger = logging.getLogger("PromoExtractor")
logger.setLevel(logging.INFO)
# Clear old handlers to prevent duplicate lines
if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)

def log_event(msg: str):
    """Write log to outputs/pipeline_run.log and print to stdout."""
    logger.info(msg)

def record_llm_interaction(
    stage: str,
    model: str,
    system_prompt: Optional[str],
    messages: List[Dict[str, str]],
    raw_response: str,
    parsed_json: Optional[Dict[str, Any]] = None,
    latency_seconds: float = 0.0
):
    """
    Logs every LLM prompt sent and response received into a single master JSON file
    in 'outputs/extras/llm_chats.json'.
    """
    try:
        os.makedirs(LLM_CHATS_DIR, exist_ok=True)
        data = {
            "project": "Context-Aware Microdrama Promo Extractor",
            "log_file": "outputs/extras/llm_chats.json",
            "last_updated": datetime.now().isoformat(),
            "total_recorded_requests": 0,
            "interactions": []
        }
        if os.path.exists(LLM_CHATS_FILE):
            try:
                with open(LLM_CHATS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict) and "interactions" in existing:
                        data = existing
            except Exception:
                pass

        interaction_entry = {
            "interaction_id": len(data["interactions"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "model": model,
            "latency_seconds": round(latency_seconds, 2),
            "request": {
                "system_prompt": system_prompt or "",
                "messages": messages
            },
            "response": {
                "raw_content": raw_response,
                "parsed_json": parsed_json
            }
        }
        data["interactions"].append(interaction_entry)
        data["total_recorded_requests"] = len(data["interactions"])
        data["last_updated"] = datetime.now().isoformat()

        with open(LLM_CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log_event(f"[LLM Chat Logger] Failed to record interaction: {e}")


# Disable remote network calls on startup for model pricing and telemetry
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"

try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.telemetry = False
except ImportError:
    litellm = None


class APIKeyPool:
    """
    Manages a pool of Gemini / LLM API keys with automatic failover,
    round-robin rotation, and 429 rate-limit cooldown handling.
    """

    def __init__(self, raw_keys: Optional[Union[str, List[str]]] = None):
        self.keys: List[str] = []
        self.current_index = 0
        self.cooldowns: Dict[str, float] = {}  # key -> cooldown expiry timestamp
        self.set_keys(raw_keys)

    def set_keys(self, raw_keys: Optional[Union[str, List[str]]] = None):
        """Parse keys from list, comma/newline string, or discover from environment."""
        extracted = []
        if raw_keys:
            if isinstance(raw_keys, list):
                for k in raw_keys:
                    if k and isinstance(k, str) and k.strip():
                        extracted.append(k.strip().strip('"').strip("'"))
            elif isinstance(raw_keys, str):
                # Split on comma, semicolon, or newlines
                parts = re.split(r"[\r\n,;]+", raw_keys)
                for p in parts:
                    clean = p.strip().strip('"').strip("'")
                    if clean:
                        extracted.append(clean)

        # Also search environment variables for additional keys
        env_keys_str = os.environ.get("GEMINI_API_KEYS", "")
        if env_keys_str:
            for p in re.split(r"[\r\n,;]+", env_keys_str):
                clean = p.strip().strip('"').strip("'")
                if clean and clean not in extracted:
                    extracted.append(clean)

        # Check numbered keys: GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
        for i in range(1, 15):
            k = os.environ.get(f"GEMINI_API_KEY_{i}")
            if k:
                clean = k.strip().strip('"').strip("'")
                if clean and clean not in extracted:
                    extracted.append(clean)

        # Check default keys
        single_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI API KEY") or os.environ.get("GOOGLE_API_KEY")
        if single_key:
            clean = single_key.strip().strip('"').strip("'")
            if clean and clean not in extracted:
                extracted.append(clean)

        # Check other provider keys
        for env_var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY"]:
            k = os.environ.get(env_var)
            if k:
                clean = k.strip().strip('"').strip("'")
                if clean and clean not in extracted:
                    extracted.append(clean)

        self.keys = extracted
        self.current_index = 0
        log_event(f"[Key Pool] Initialized with {len(self.keys)} API key(s).")

    def get_active_key(self) -> Optional[str]:
        """Returns the current healthy key, rotating past any keys on cooldown."""
        if not self.keys:
            return None

        now = time.time()
        # Find first key that is not on cooldown
        for attempt in range(len(self.keys)):
            idx = (self.current_index + attempt) % len(self.keys)
            key = self.keys[idx]
            expiry = self.cooldowns.get(key, 0.0)
            if now >= expiry:
                self.current_index = idx
                return key

        # If all keys are on cooldown, return the one that expires soonest
        soonest_key = min(self.keys, key=lambda k: self.cooldowns.get(k, 0.0))
        wait_seconds = max(0.5, self.cooldowns.get(soonest_key, 0.0) - now)
        log_event(f"[Key Pool] All {len(self.keys)} keys are cooling down. Waiting {wait_seconds:.1f}s for key to reset...")
        time.sleep(min(wait_seconds, 15.0))
        return soonest_key

    def rotate_on_rate_limit(self, failed_key: Optional[str], cooldown_seconds: float = 60.0) -> Optional[str]:
        """Flag the failed key as rate-limited and rotate immediately to the next key."""
        if failed_key:
            self.cooldowns[failed_key] = time.time() + cooldown_seconds
            mask = f"{failed_key[:6]}...{failed_key[-4:]}" if len(failed_key) > 10 else "KEY"
            log_event(f"[Key Pool] Rate limit on {mask}. Cooldown set for {cooldown_seconds}s.")

        if len(self.keys) > 1:
            self.current_index = (self.current_index + 1) % len(self.keys)
            next_key = self.get_active_key()
            mask_next = f"{next_key[:6]}...{next_key[-4:]}" if next_key and len(next_key) > 10 else "KEY"
            log_event(f"[Key Pool] Rotated to Key #{self.current_index + 1}/{len(self.keys)} ({mask_next}).")
            return next_key
        else:
            log_event("[Key Pool] Only 1 key available. Pausing before retry...")
            time.sleep(3.0)
            return failed_key

    def count(self) -> int:
        return len(self.keys)


def resolve_api_keys(raw_keys: Optional[Union[str, List[str]]] = None) -> List[str]:
    """Helper function to discover and resolve all available API keys."""
    pool = APIKeyPool(raw_keys)
    return pool.keys



class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Providers to enable modular swappability."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text completion from messages."""
        pass

    @abstractmethod
    def generate_json(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate structured JSON completion adhering to enforced JSON output."""
        pass


class LiteLLMProvider(BaseLLMProvider):
    """
    Production-grade LLM Provider using litellm.completion().
    Features:
    - Dynamic Model Routing (per-stage or global override)
    - Multi-Key Rotation Pool (supports Gemini, OpenAI, Anthropic keys)
    - Automatic Failover on 429 (Too Many Requests) & 503 (Service Unavailable)
    - Adaptive Request Pacing (sleep interval to avoid 15 RPM rate limits)
    - Autonomous JSON repair & validation
    """

    DEFAULT_ROUTING = {
        "Macro Narrative Director Skim": "gemini/gemini-1.5-flash",
        "Precision Concept Editor": "gemini/gemini-1.5-pro",
        "Executive Cohesion Ranker": "gemini/gemini-1.5-pro"
    }

    FALLBACK_MODELS = [
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-flash-lite-latest",
        "gemini/gemini-3-flash-preview",
        "gemini/gemini-flash-latest",
        "gemini/gemini-1.5-flash"
    ]

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        pacing_delay: float = 2.5,
        **kwargs
    ):
        self.DEFAULT_ROUTING = LiteLLMProvider.DEFAULT_ROUTING.copy()
        self.custom_routing: Dict[str, str] = {}
        self.raw_model_name = model
        self.model = self._normalize_model_name(model) if model else self.DEFAULT_ROUTING["Macro Narrative Director Skim"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pacing_delay = pacing_delay
        self.extra_kwargs = kwargs
        
        # Initialize key pool
        self.key_pool = APIKeyPool(api_key)

    def _match_stage(self, stage_name: str) -> str:
        """Helper to match exact or partial stage strings to standard routing keys."""
        if not stage_name:
            return "Macro Narrative Director Skim"
        stage_clean = str(stage_name).strip()
        for known_stage in self.DEFAULT_ROUTING.keys():
            if known_stage.lower() in stage_clean.lower() or stage_clean.lower().startswith(known_stage.lower()):
                return known_stage
        return stage_clean

    def set_custom_route(self, stage: str, model: Optional[str] = None):
        """Update custom routing for a specific stage. If model is empty/None, clear that stage's override."""
        matched_stage = self._match_stage(stage)
        if not model or not str(model).strip():
            self.custom_routing.pop(matched_stage, None)
            log_event(f"[Model Routing] Cleared custom override for stage: '{matched_stage}'")
        else:
            norm_model = self._normalize_model_name(str(model).strip())
            self.custom_routing[matched_stage] = norm_model
            log_event(f"[Model Routing] Set custom route for '{matched_stage}' -> {norm_model}")

    def set_global_override(self, model: Optional[str] = None):
        """Sets the custom route for all three stages simultaneously, or clears all overrides if model is empty/None."""
        if not model or not str(model).strip():
            self.custom_routing.clear()
            log_event("[Model Routing] Cleared all global overrides. Reverting to default routing.")
        else:
            norm_model = self._normalize_model_name(str(model).strip())
            for stage in self.DEFAULT_ROUTING.keys():
                self.custom_routing[stage] = norm_model
            log_event(f"[Model Routing] Set global model override across all stages -> {norm_model}")

    def _normalize_model_name(self, model: str) -> str:
        """Ensure model string matches litellm conventions and preserves external providers."""
        if not model:
            return "gemini/gemini-1.5-flash"
        model = model.strip()
        clean = model.lower()

        # If already provider-prefixed or third-party (OpenAI, Anthropic, DeepSeek, Mistral)
        if (
            clean.startswith("gpt") or clean.startswith("openai/") or
            clean.startswith("claude") or clean.startswith("anthropic/") or
            clean.startswith("deepseek") or clean.startswith("mistral") or
            clean.startswith("o1") or clean.startswith("o3")
        ):
            return model

        # Gemini model mapping
        if model in ["gemini-1.5-flash", "gemini/gemini-1.5-flash", "gemini-flash"]:
            return "gemini/gemini-1.5-flash"
        elif model in ["gemini-1.5-pro", "gemini/gemini-1.5-pro", "gemini-pro"]:
            return "gemini/gemini-1.5-pro"
        elif model.startswith("gemini-") and not model.startswith("gemini/"):
            return f"gemini/{model}"
        return model

    def _inject_api_key_for_model(self, model_name: str, key: Optional[str]):
        """Dynamically injects active API key into environment variables based on model prefix."""
        if not key:
            return
        clean = model_name.lower()
        if "gpt" in clean or "openai" in clean or "o1" in clean or "o3" in clean:
            os.environ["OPENAI_API_KEY"] = key
        elif "claude" in clean or "anthropic" in clean:
            os.environ["ANTHROPIC_API_KEY"] = key
        elif "gemini" in clean or "google" in clean:
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key
        elif "mistral" in clean:
            os.environ["MISTRAL_API_KEY"] = key
        elif "deepseek" in clean:
            os.environ["DEEPSEEK_API_KEY"] = key
        else:
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key

    def set_model(self, model: str):
        self.raw_model_name = model
        self.model = self._normalize_model_name(model)
        self.set_global_override(self.model)
        log_event(f"[Provider] Model updated to: {self.model}")

    def set_api_keys(self, keys: Union[str, List[str]]):
        """Update API key pool dynamically from UI or config."""
        self.key_pool.set_keys(keys)

    def set_pacing_delay(self, delay_seconds: float):
        """Configure sleep interval between successive calls to avoid 15 RPM rate limits."""
        self.pacing_delay = max(0.0, float(delay_seconds))

    def pace(self):
        """Applies sleep pacing between successive LLM requests."""
        if self.pacing_delay > 0:
            time.sleep(self.pacing_delay)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if litellm is None:
            raise RuntimeError("LiteLLM is not installed.")
            
        self.pace()
        key = self.key_pool.get_active_key()
        stage_raw = kwargs.get("stage", "")
        stage = self._match_stage(stage_raw)
        active_model = self.custom_routing.get(stage) or self.DEFAULT_ROUTING.get(stage, self.model)
        active_model = self._normalize_model_name(active_model)

        if key:
            self._inject_api_key_for_model(active_model, key)

        params = {
            "model": active_model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "timeout": kwargs.get("timeout", 30),
            **self.extra_kwargs
        }
        if key:
            params["api_key"] = key

        response = litellm.completion(**params)
        return response.choices[0].message.content

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates structured JSON with dynamic Model Routing, multi-key failover on 429/503 errors,
        and autonomous JSON syntax repair.
        """
        if litellm is None:
            raise RuntimeError("LiteLLM is not installed.")

        # Apply pacing delay to prevent hitting 15 RPM rate limits
        self.pace()

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # 1. Determine current stage and primary model
        stage_raw = kwargs.get("stage", "")
        stage = self._match_stage(stage_raw)
        is_custom = stage in self.custom_routing and bool(self.custom_routing[stage])

        primary_model = self.custom_routing.get(stage) or self.DEFAULT_ROUTING.get(stage, "gemini/gemini-1.5-flash")
        primary_model = self._normalize_model_name(primary_model)

        # 2. Build candidate models list dynamically with fallbacks
        candidate_models = [primary_model]

        # If primary_model is a custom override, add self.DEFAULT_ROUTING.get(stage) as immediate first fallback
        if is_custom:
            default_stage_model = self._normalize_model_name(self.DEFAULT_ROUTING.get(stage, "gemini/gemini-1.5-flash"))
            if default_stage_model not in candidate_models:
                candidate_models.append(default_stage_model)

        # Then append general free-tier fallback models
        for fb in self.FALLBACK_MODELS:
            norm_fb = self._normalize_model_name(fb)
            if norm_fb not in candidate_models:
                candidate_models.append(norm_fb)

        total_key_count = max(1, self.key_pool.count())
        max_total_attempts = max(max_retries, total_key_count * 2, len(candidate_models))
        last_error = None

        for attempt in range(max_total_attempts):
            active_key = self.key_pool.get_active_key()

            # Pick model (rotate if previous failed)
            current_model = candidate_models[attempt % len(candidate_models)]

            # Dynamically inject active API key into environment based on model prefix
            if active_key:
                self._inject_api_key_for_model(current_model, active_key)

            params = {
                "model": current_model,
                "messages": full_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "timeout": kwargs.get("timeout", 30),
                "response_format": {"type": "json_object"},
                **self.extra_kwargs
            }
            if active_key:
                params["api_key"] = active_key

            try:
                t0 = time.time()
                response = litellm.completion(**params)
                dt = time.time() - t0
                content = response.choices[0].message.content or ""
                parsed = self._parse_and_clean_json(content)
                log_event(f"[LLM Success] Stage '{stage}' | Model {current_model} returned valid JSON in {dt:.2f}s.")
                record_llm_interaction(
                    stage=stage_raw or stage,
                    model=current_model,
                    system_prompt=system_prompt,
                    messages=messages,
                    raw_response=content,
                    parsed_json=parsed,
                    latency_seconds=dt
                )
                return parsed

            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                # Handle 429 Too Many Requests / Quota Exceeded
                if "429" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    log_event(f"[Rate Limit 429] Encountered on model {current_model}: {str(e)[:120]}")
                    # Rotate key immediately
                    self.key_pool.rotate_on_rate_limit(active_key, cooldown_seconds=60.0)
                    time.sleep(2.0)
                    continue

                # Handle 503 Service Unavailable / Server Overload / 404 Model Not Found
                elif "503" in err_str or "unavailable" in err_str or "high demand" in err_str or "404" in err_str or "not found" in err_str:
                    log_event(f"[Model Fallback] {current_model} unavailable ({str(e)[:80]}). Cascading to next candidate...")
                    time.sleep(2.0)
                    continue

                # JSON parsing error: prompt LLM to self-correct
                elif "json" in err_str or isinstance(e, (ValueError, json.JSONDecodeError)):
                    log_event(f"[JSON Parse Error] Attempt {attempt+1}: {str(e)[:100]}. Requesting clean JSON repair...")
                    full_messages.append({
                        "role": "user",
                        "content": f"Your previous response had a JSON parsing issue: {str(e)}. Please output ONLY valid, strictly well-formed JSON without markdown or trailing commas."
                    })
                    time.sleep(1.0)
                    continue

                else:
                    log_event(f"[Provider Error] Model {current_model} attempt {attempt+1}: {str(e)[:140]}")
                    time.sleep(2.0)

        raise RuntimeError(f"Failed to generate valid JSON for stage '{stage}' after {max_total_attempts} attempts. Last error: {last_error}")

    def _parse_and_clean_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Strips markdown code blocks, locates outermost curly brackets,
        and safely parses JSON even if slightly malformed.
        """
        text = raw_text.strip()
        
        # 1. Direct parse attempt
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Strip ```json ... ``` code fence
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass

        # 3. Locate outermost { ... }
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            extracted = text[start_idx:end_idx + 1]
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                # 4. Remove trailing commas before } or ]
                sanitized = re.sub(r",\s*([}\]])", r"\1", extracted)
                try:
                    return json.loads(sanitized)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Could not parse valid JSON from LLM output:\n{raw_text}")
