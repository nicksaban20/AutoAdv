import datetime
import colorama
from colorama import Fore, Style, Back
import os

# Initialize colorama for cross-platform colored terminal text
colorama.init(autoreset=True)

# Verbosity levels (will be imported from config.py in practice)
VERBOSE_NONE = 0  # Only show critical information
VERBOSE_NORMAL = 1  # Show important processes
VERBOSE_DETAILED = 2  # Show all details

# Global verbosity setting (imported from config.py in practice)
VERBOSE_LEVEL = VERBOSE_NORMAL

# Map verbosity levels to names for display
VERBOSE_LEVEL_NAMES = {
    VERBOSE_NONE: "Minimal",
    VERBOSE_NORMAL: "Normal",
    VERBOSE_DETAILED: "Detailed",
}

def log(message, type="info", verbose_level=VERBOSE_NORMAL):
    """
    Enhanced logging function with color coding and verbosity control.
    
    Args:
        message (str): The message to log
        type (str): The type of message (info, success, error, warning, debug, config, result)
        verbose_level (int): The verbosity level of this message
    """
    # Import here to avoid circular imports
    from config import VERBOSE_LEVEL
    
    # Skip if verbosity level is too low
    if verbose_level > VERBOSE_LEVEL:
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Color coding based on message type
    if type == "info":
        prefix = f"{Fore.BLUE}[INFO]{Style.RESET_ALL}"
    elif type == "success":
        prefix = f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL}"
    elif type == "error":
        prefix = f"{Fore.RED}[ERROR]{Style.RESET_ALL}"
    elif type == "warning":
        prefix = f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}"
    elif type == "debug":
        prefix = f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL}"
    elif type == "config":
        prefix = f"{Fore.CYAN}[CONFIG]{Style.RESET_ALL}"
    elif type == "result":
        prefix = f"{Back.BLUE}{Fore.WHITE}[RESULT]{Style.RESET_ALL}"
    else:
        prefix = f"[{type.upper()}]"

    # Print formatted message
    print(f"{prefix} {timestamp} | {message}")
    
def print_header(title, width=60):
    """
    Print a formatted header with a title.
    
    Args:
        title (str): The title of the header
        width (int): The width of the header bar
    """
    print(f"\n{Fore.CYAN}{'=' * width}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title.center(width)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * width}{Style.RESET_ALL}\n")
    
def print_section(title, width=60):
    """
    Print a formatted section header.
    
    Args:
        title (str): The title of the section
        width (int): The width of the section bar
    """
    print(f"\n{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * width}{Style.RESET_ALL}")
    
def display_config(config_dict, width=80):
    """
    Display configuration in a clean, structured format.
    
    Args:
        config_dict (dict): Dictionary containing configuration parameters
        width (int): Width of the display
    """
    print_header("CONFIGURATION", width)

    # Display model information
    print(
        f"{Fore.CYAN}TARGET MODEL:{Style.RESET_ALL} {config_dict['target_model']} - "
        f"{config_dict.get('target_model_name', 'Unknown')}"
    )
    print(f"{Fore.CYAN}ATTACKER MODEL:{Style.RESET_ALL} {config_dict.get('attacker_model', 'gpt-4o-mini')}")

    print_section("PARAMETERS")
    print(f"  • Attacker Temperature: {config_dict['attacker_temp']}")
    print(f"  • Target Temperature: {config_dict['target_temp']}")
    print(f"  • Number of Turns: {config_dict['turns']}")
    print(f"  • StrongReject Threshold: {config_dict['strongreject_threshold']}")
    print(
        f"  • Target Memory Enabled: {'Yes' if config_dict.get('target_memory_enabled', False) else 'No'}"
    )

    print_section("EXECUTION SETTINGS")
    print(f"  • Sample Size: {config_dict.get('sample_size', 'All prompts')}")
    print(f"  • Parallel Workers: {config_dict.get('max_workers', 1)}")
    print(f"  • Verbosity Level: {VERBOSE_LEVEL_NAMES.get(config_dict.get('verbosity_level', VERBOSE_NORMAL), 'Detailed')}")

    print_section("FILE PATHS")
    print(f"  • Prompts: {config_dict.get('adversarial_prompts', 'Not specified')}")
    print(f"  • System Prompt: {config_dict.get('system_prompt', 'Not specified')}")
    print(
        f"  • Followup Prompt: {config_dict.get('system_prompt_followup', 'Not Used')}"
    )

    print(f"{Fore.CYAN}{'=' * width}{Style.RESET_ALL}\n")

def ensure_directory_exists(directory):
    """
    Make sure a directory exists, creating it if necessary
    
    Args:
        directory (str): Path to directory to ensure exists
        
    Returns:
        str: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        log(f"Created directory: {directory}", "info", VERBOSE_DETAILED)
    return directory