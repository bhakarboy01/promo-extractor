import os
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

# Automatically load environment variables from current directory and Test folder
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "Test", ".env"))

def resolve_api_keys():
    """Ensure environment variables like GEMINI_API_KEY are properly formatted."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI API KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        clean_key = gemini_key.strip().strip('"').strip("'")
        os.environ["GEMINI_API_KEY"] = clean_key
        os.environ["GOOGLE_API_KEY"] = clean_key

resolve_api_keys()

# Disable remote network calls on startup for model pricing and telemetry
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"

try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.telemetry = False
except ImportError:
    litellm = None


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
    Modular LLM Provider using litellm.completion().
    Default model: gemini-1.5-flash (mapped to high-performance active Gemini Flash Lite).
    Enforces JSON response_format and provides autonomous parsing, repair & multi-model fallback.
    """

    FALLBACK_MODELS = [
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-flash-lite-latest",
        "gemini/gemini-3-flash-preview",
        "gemini/gemini-flash-latest"
    ]

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs
    ):
        self.raw_model_name = model
        self.model = self._normalize_model_name(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs
        
        if api_key:
            self.api_key = api_key.strip().strip('"').strip("'")
            if "gemini" in self.model:
                os.environ["GEMINI_API_KEY"] = self.api_key
                os.environ["GOOGLE_API_KEY"] = self.api_key
            elif "gpt" in self.model:
                os.environ["OPENAI_API_KEY"] = self.api_key
            elif "claude" in self.model:
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
        else:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def _normalize_model_name(self, model: str) -> str:
        """Ensure model string matches litellm conventions and maps retired endpoints."""
        model = model.strip()
        # Google AI Studio mappings for optimal speed & reliability
        if model in ["gemini-1.5-flash", "gemini/gemini-1.5-flash", "gemini-flash"]:
            return "gemini/gemini-flash-lite-latest"
        elif model in ["gemini-1.5-pro", "gemini/gemini-1.5-pro"]:
            return "gemini/gemini-3.1-pro-preview"
        elif model.startswith("gemini-") and not model.startswith("gemini/"):
            return f"gemini/{model}"
        return model

    def set_model(self, model: str):
        self.raw_model_name = model
        self.model = self._normalize_model_name(model)

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip().strip('"').strip("'")
        if "gemini" in self.model:
            os.environ["GEMINI_API_KEY"] = self.api_key
            os.environ["GOOGLE_API_KEY"] = self.api_key

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if litellm is None:
            raise RuntimeError("LiteLLM is not installed. Please run `pip install litellm`.")
            
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **self.extra_kwargs
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = litellm.completion(**params)
        return response.choices[0].message.content

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_retries: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response.
        Enforces response_format={'type': 'json_object'} and performs autonomous cleaning & fallback.
        """
        if litellm is None:
            raise RuntimeError("LiteLLM is not installed. Please run `pip install litellm`.")

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        candidate_models = [self.model]
        if "gemini" in self.model:
            for fb in self.FALLBACK_MODELS:
                if fb not in candidate_models:
                    candidate_models.append(fb)

        last_error = None
        for current_model in candidate_models:
            params = {
                "model": current_model,
                "messages": full_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "timeout": kwargs.get("timeout", 25),
                "response_format": {"type": "json_object"},
                **self.extra_kwargs
            }
            if self.api_key:
                params["api_key"] = self.api_key

            for attempt in range(max_retries + 1):
                try:
                    response = litellm.completion(**params)
                    content = response.choices[0].message.content or ""
                    parsed = self._parse_and_clean_json(content)
                    return parsed
                except Exception as e:
                    last_error = e
                    # If it's a 503 or 404 from Google, immediately try next candidate model
                    err_str = str(e).lower()
                    if "503" in err_str or "404" in err_str or "unavailable" in err_str or "high demand" in err_str:
                        break
                    
                    if attempt < max_retries:
                        full_messages.append({
                            "role": "user",
                            "content": f"Your previous response had a JSON parsing issue: {str(e)}. Please output ONLY valid, strictly well-formed JSON without markdown or trailing commas."
                        })
                        params["messages"] = full_messages

        raise RuntimeError(f"Failed to generate valid JSON across candidate models. Last Error: {last_error}")

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
