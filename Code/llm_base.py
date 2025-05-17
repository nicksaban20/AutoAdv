from token_calculator import TokenCalculator

class LLM:
    """
    Base class for language model functionality. Provides common methods and attributes
    for both attacker and target models.
    
    Attributes:
        model (str): Name of the language model
        temperature (float): Temperature setting for model generation
        history (list): Conversation history as a list of message dictionaries
        tokenCalculator (TokenCalculator): Utility for token counting and cost calculation
    """
    
    def __init__(self, model, temperature, requestCostPerToken, responseCostPerToken, tokenModel=None):
        """
        Initialize a new language model instance.
        
        Args:
            model (str): The model identifier
            temperature (float): Controls randomness in generation (0.0-2.0)
            requestCostPerToken (float): Cost per million tokens for requests
            responseCostPerToken (float): Cost per million tokens for responses
            tokenModel (str, optional): Model name for tokenization. Defaults to model name.
        """
        self.model = model
        self.temperature = temperature
        self.history = []

        # model token costs
        self.tokenCalculator = TokenCalculator(
            requestCostPerToken, responseCostPerToken, tokenModel or model
        )

    def append_to_history(self, role, message):
        """
        Add a message to conversation history
        
        Args:
            role (str): The role of the message sender (system, user, assistant)
            message (str or list): Content of the message
        """
        self.history.append({"role": role, "content": message})
    
    def clear_history(self):
        """Clear conversation history while preserving system messages"""
        # Keep only system messages
        self.history = [msg for msg in self.history if msg["role"] == "system"]
    
    def get_last_message(self, role=None):
        """
        Get the last message in history, optionally filtering by role
        
        Args:
            role (str, optional): Filter by role (system, user, assistant)
            
        Returns:
            dict or None: The last message matching the criteria or None
        """
        if not self.history:
            return None
            
        if role:
            # Find the last message with the specified role
            for msg in reversed(self.history):
                if msg["role"] == role:
                    return msg
            return None
        else:
            # Return the last message regardless of role
            return self.history[-1]
            
    def calculate_history_tokens(self):
        """
        Calculate total tokens used in conversation history
        
        Returns:
            int: Total token count
        """
        return sum(
            self.tokenCalculator.calculate_tokens(
                msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            )
            for msg in self.history
        )