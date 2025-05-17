# Make sure OpenAI is imported if used
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    pass
# Make sure Together is imported if used
try:
    from together import Together
except ImportError:
    Together = None
    pass
# Make sure Anthropic is imported if used
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
    pass

from llm_base import LLM
from utils import check_api_key_existence, api_call_with_retry
from logging_utils import log
from config import TARGET_MODELS, VERBOSE_DETAILED


class TargetLLM(LLM):
    """
    Specialized LLM class for the target model that will be tested for safety
    guardrail effectiveness.

    Attributes:
        model_key (str): Key identifier for the target model
        api_type (str): API provider type ("together", "openai", "anthropic")
        memory_enabled (bool): Whether the model maintains conversation history
        client: The API client instance
    """

    def __init__(
        self,
        temperature,
        target_model_key="llama3-8b",
        memory_enabled=False,
        model_config=None,
    ):
        """
        Initialize a target model instance.

        Args:
            temperature (float): Controls randomness in generation (0.0-2.0)
            target_model_key (str): Key for the model in TARGET_MODELS dictionary
            memory_enabled (bool): Whether to use full conversation history
            model_config (dict, optional): Model configuration if not using predefined models
        """
        if model_config:
            # Use provided configuration
            super().__init__(
                model=model_config["name"],
                temperature=temperature,
                requestCostPerToken=model_config["request_cost"],
                responseCostPerToken=model_config["response_cost"],
                tokenModel=model_config.get("token_model"),
            )
            self.model_key = target_model_key
            self.api_type = model_config.get(
                "api", "together"
            )  # Default to together if not specified
        else:
            # Use model_key from caller
            if target_model_key not in TARGET_MODELS:
                raise ValueError(
                    f"Unknown target model: {target_model_key}. Available options: {', '.join(TARGET_MODELS.keys())}"
                )

            model_config = TARGET_MODELS[target_model_key]
            self.model_key = target_model_key
            self.api_type = model_config.get("api", "together")  # Default if missing

            super().__init__(
                model=model_config["name"],
                temperature=temperature,
                requestCostPerToken=model_config["request_cost"],
                responseCostPerToken=model_config["response_cost"],
                tokenModel=model_config.get("token_model"),  # Use get for safety
            )

        self.memory_enabled = memory_enabled
        self.client = self._initialize_api_client()  # Use helper method
        # Ensure history is initialized for target
        self.history = []

    def _initialize_api_client(self):
        """Initialize the appropriate API client based on model provider"""
        log(
            f"Initializing target client for API type: {self.api_type}",
            "debug",
            VERBOSE_DETAILED,
        )
        if self.api_type == "together":
            # Together uses OpenAI client format
            if not OpenAI:
                raise ImportError("OpenAI library not installed (needed for Together).")
            return OpenAI(
                api_key=check_api_key_existence("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        elif self.api_type == "openai":
            if not OpenAI:
                raise ImportError("OpenAI library not installed.")
            return OpenAI(api_key=check_api_key_existence("OPENAI_API_KEY"))
        elif self.api_type == "anthropic":
            if not Anthropic:
                raise ImportError(
                    "Anthropic library not installed. pip install anthropic"
                )
            return Anthropic(api_key=check_api_key_existence("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported API type for target: {self.api_type}")

    def converse(self, request):
        """
        Generate a response from the target model based on the provided request.

        Args:
            request (str): The request/prompt to send to the model

        Returns:
            tuple: (response, request_tokens, response_tokens, request_cost, response_cost)
                   Returns (None, 0, 0, 0, 0) on error.
        """
        if not request:
            log("Target converse called with empty request.", "warning")
            return None, 0, 0, 0.0, 0.0

        try:
            # --- History Management ---
            messages_to_send = []
            if self.memory_enabled:
                # Ensure the new user request is added correctly
                # If history is empty or last message was assistant, add user message
                if not self.history or self.history[-1].get("role") == "assistant":
                    self.append_to_history("user", request)
                # If last message was user, OVERWRITE it with the new request
                elif self.history[-1].get("role") == "user":
                    log(
                        "Overwriting last user message in history with new request.",
                        "debug",
                        VERBOSE_DETAILED + 1,
                    )
                    self.history[-1]["content"] = request
                # If history is weird (e.g. system, system), just add user message
                else:
                    self.append_to_history("user", request)

                messages_to_send = self.history
                log(
                    f"Target ({self.model}) using stateful conversation. History length: {len(messages_to_send)}",
                    "debug",
                    VERBOSE_DETAILED,
                )
                # log(f"Target history being sent: {messages_to_send}", "debug", VERBOSE_DETAILED+1) # Very verbose
            else:
                # Stateless: only send the current request as user
                messages_to_send = [{"role": "user", "content": request}]
                log(
                    f"Target ({self.model}) using stateless conversation",
                    "debug",
                    VERBOSE_DETAILED,
                )

            # --- API Call Preparation ---
            api_args = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": 1024,  # Reasonable default limit
            }
            response_content = None
            completion_details = None  # To store usage info if available directly

            # --- API Call Logic ---
            if self.api_type in ["together", "openai"]:
                api_args["messages"] = messages_to_send

                completion = api_call_with_retry(
                    self.client.chat.completions.create, **api_args  # Pass the method
                )

                if completion.choices:
                    response_content = completion.choices[0].message.content

                completion_details = completion  # Keep for potential usage info

            elif self.api_type == "anthropic":
                # Format messages for Anthropic (handle system prompt if needed)
                anthropic_messages = []
                system_prompt_content = None
                for msg in messages_to_send:
                    # Anthropic expects system prompt separately
                    if msg["role"] == "system":
                        if "system" not in api_args:  # Anthropic specific parameter
                            api_args["system"] = msg["content"]
                        else:  # Append if multiple system prompts (though unusual)
                            api_args["system"] += "\n" + msg["content"]
                        continue  # Don't add system prompts to messages list

                    # Basic role mapping, ensure user/assistant alternation if possible
                    role = "user" if msg["role"] == "user" else "assistant"
                    anthropic_messages.append({"role": role, "content": msg["content"]})

                # Validate message alternation for Anthropic if needed (API might enforce this)
                # ... validation logic ...

                api_args["messages"] = anthropic_messages  # Add formatted messages

                completion = self.client.messages.create(
                    **api_args  # Pass prepared args
                )
                if completion.content:
                    response_content = completion.content[0].text
                completion_details = completion  # Keep for potential usage info

            # --- Response Handling ---
            if response_content is None:
                log(
                    f"No response content received from target model {self.model}. API response: {completion_details}",
                    "error",
                )
                response_content = "[Model returned no content]"  # Ensure it's a string

            if not isinstance(response_content, str) or response_content.strip() == "":
                log(
                    f"Empty or non-string response received from target model {self.model}: {response_content}",
                    "warning",
                )
                response_content = "[Model returned empty or invalid response]"

            response_content = response_content.strip()

            # Add valid response to history *if* memory is enabled
            if self.memory_enabled:
                # Ensure assistant message follows a user message
                if self.history and self.history[-1].get("role") == "user":
                    self.append_to_history("assistant", response_content)
                else:
                    log(
                        f"Target memory enabled, but history state prevents adding assistant message. Last: {self.history[-1] if self.history else 'Empty'}",
                        "warning",
                        VERBOSE_DETAILED,
                    )

            # --- Token Calculation ---
            # Request tokens calculation needs to reflect what was actually sent
            request_text_for_calc = ""
            if self.api_type == "anthropic" and "system" in api_args:
                request_text_for_calc = (
                    api_args["system"]
                    + "\n"
                    + "\n".join([msg["content"] for msg in api_args["messages"]])
                )
            elif "messages" in api_args:
                request_text_for_calc = "\n".join(
                    [msg["content"] for msg in api_args["messages"]]
                )

            requestTokens = self.tokenCalculator.calculate_tokens(request_text_for_calc)
            responseTokens = self.tokenCalculator.calculate_tokens(response_content)

            # Try to get exact usage from completion details if available (more accurate)
            if (
                completion_details
                and hasattr(completion_details, "usage")
                and completion_details.usage
            ):
                usage_info = completion_details.usage
                prompt_tokens_api = getattr(
                    usage_info,
                    "input_tokens",
                    getattr(usage_info, "prompt_tokens", None),
                )
                completion_tokens_api = getattr(
                    usage_info,
                    "output_tokens",
                    getattr(usage_info, "completion_tokens", None),
                )

                if prompt_tokens_api is not None:
                    requestTokens = prompt_tokens_api
                if completion_tokens_api is not None:
                    responseTokens = completion_tokens_api
                log(
                    f"Using exact token counts from API response: Req={requestTokens}, Resp={responseTokens}",
                    "debug",
                    VERBOSE_DETAILED + 1,
                )

            requestCost = self.tokenCalculator.calculate_cost(
                requestTokens, isRequest=True
            )
            responseCost = self.tokenCalculator.calculate_cost(
                responseTokens, isRequest=False
            )

            log(
                f"Target ({self.model}) responded. Req Tokens: {requestTokens}, Resp Tokens: {responseTokens}",
                "debug",
                VERBOSE_DETAILED,
            )

            return (
                response_content,
                requestTokens,
                responseTokens,
                requestCost,
                responseCost,
            )

        except Exception as e:
            log(
                f"An error occurred when conversing with the target ({self.model}): {e}",
                "error",
            )
            import traceback

            log(traceback.format_exc(), "error", VERBOSE_DETAILED)
            # Clean up history potentially if memory is enabled and call failed mid-way?
            # For now, just return failure state.
            return None, 0, 0, 0.0, 0.0  # failure values
