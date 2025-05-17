import os
import sys
import random
import time
import concurrent.futures
import argparse
import pandas as pd
from tqdm import tqdm

# Add Helpers directory to path
sys.path.insert(
    1,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Helpers"
    ),
)

# Import modules
from config import (
    TARGET_MODELS,
    DEFAULT_PATHS,
    DEFAULT_CONFIG,
    VERBOSE_LEVEL,
    VERBOSE_NORMAL,
    VERBOSE_DETAILED,
    VERBOSE_NONE,
    VERBOSE_LEVEL_NAMES,
)
from logging_utils import log, display_config, ensure_directory_exists
from utils import is_model_available, get_next_filename
from attacker_llm import AttackerLLM
from target_llm import TargetLLM
from conversation import multi_turn_conversation, save_conversation_log
from pattern_manager import PatternManager


def load_prompts(filepath, sample_size=None):
    """
    Load prompts from CSV file with optional sampling.

    Args:
        filepath (str): Path to the CSV file
        sample_size (int, optional): Number of prompts to randomly sample

    Returns:
        list: List of prompts
    """
    log(f"Loading prompts from {filepath}", "info")

    try:
        # Read CSV file
        df_prompts = pd.read_csv(filepath)
        all_prompts = df_prompts["prompt"].tolist()

        # Handle sampling
        if sample_size and sample_size < len(all_prompts):
            prompts = random.sample(all_prompts, sample_size)
            log(
                f"Randomly selected {sample_size} prompts out of {len(all_prompts)}",
                "info",
            )
        else:
            prompts = all_prompts
            if sample_size:
                log(
                    f"Sample size {sample_size} >= total prompts {len(all_prompts)}. Using all prompts.",
                    "info",
                )

        return prompts
    except Exception as e:
        log(f"Error loading prompts: {e}", "error")
        return []


def load_system_prompts(initial_prompt_path, followup_prompt_path=None):
    """
    Load system prompts from files.

    Args:
        initial_prompt_path (str): Path to the initial system prompt
        followup_prompt_path (str, optional): Path to the followup system prompt

    Returns:
        tuple: (initial_prompt, followup_prompt)
    """
    # Load initial prompt
    try:
        with open(initial_prompt_path, "r") as f:
            initial_prompt = f.read()
    except Exception as e:
        log(f"Error loading initial system prompt: {e}", "error")
        initial_prompt = None

    # Load followup prompt if available
    followup_prompt = None
    if followup_prompt_path and os.path.exists(followup_prompt_path):
        try:
            with open(followup_prompt_path, "r") as f:
                followup_prompt = f.read()
        except Exception as e:
            log(f"Error loading followup system prompt: {e}", "error")

    return initial_prompt, followup_prompt


def process_prompt(prompt, config, pattern_manager=None):
    """
    Process a single prompt through the multi-turn conversation framework.

    Args:
        prompt (str): The prompt to process
        config (dict): Configuration dictionary
        pattern_manager (PatternManager, optional): Pattern manager for tracking successful patterns

    Returns:
        tuple: (conversation_log, is_success)
    """
    # Create attacker and target models
    attacker = AttackerLLM(
        temperature=config["attacker_temp"],
        instructions=config["initial_prompt"],
        followup_instructions=config["followup_prompt"],
        attacker_model_key=config["attacker_model"],
    )

    target = TargetLLM(
        temperature=config["target_temp"],
        target_model_key=config["target_model"],
        memory_enabled=config["target_memory_enabled"],
    )

    # Process prompt
    prompt_summary = prompt[:30] + "..." if len(prompt) > 30 else prompt
    log(f"Processing: {prompt_summary}", "info", VERBOSE_DETAILED)

    try:
        # Ensure the malicious prompt is not reframed before attacker turn 1
        conversation_log = multi_turn_conversation(
            attacker,
            target,
            prompt,  # Pass the original malicious prompt directly
            config["turns"],
            config["strongreject_threshold"],
            pattern_manager,
        )

        # Check if successful
        is_success = conversation_log.get("status") == "success"

        # Store metadata in the log for pattern analysis
        conversation_log["targetModel"] = target.model_key
        conversation_log["attackerTemp"] = attacker.temperature

        # Log success or failure
        if is_success:
            log(
                f"✅ Success! [{conversation_log.get('processing_time', 0):.2f}s] Prompt: {prompt_summary}",
                "success",
            )
        else:
            log(
                f"❌ Failed. [{conversation_log.get('processing_time', 0):.2f}s] Prompt: {prompt_summary}",
                "info",
                VERBOSE_DETAILED,
            )

        return conversation_log, is_success
    except Exception as e:
        import traceback

        log(f"Error processing '{prompt_summary}': {e}", "error")
        log(traceback.format_exc(), "error", VERBOSE_DETAILED)
        return {
            "maliciousPrompt": prompt,
            "turns": [],
            "status": "error",
            "processing_time": 0,
            "error": str(e),
        }, False


def run_experiment(config):
    """
    Run the experiment with the specified configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (conversation_logs, success_rate)
    """
    # Start timing
    start_time = time.time()

    # Initialize pattern memory
    pattern_memory = (
        PatternManager() if config.get("use_pattern_memory", True) else None
    )

    # Load prompts
    prompts = load_prompts(config["adversarial_prompts"], config["sample_size"])
    if not prompts:
        log("No prompts to process.", "error")
        return [], 0

    # Process prompts
    conversation_logs = []
    successes = 0
    total = len(prompts)

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config["max_workers"]
    ) as executor:
        # Submit tasks
        future_to_prompt = {
            executor.submit(process_prompt, prompt, config, pattern_memory): prompt
            for prompt in prompts
        }

        # Process results as they complete
        with tqdm(total=total, desc="Processing prompts") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_prompt):
                try:
                    conv_log, is_success = future.result()
                    conversation_logs.append(conv_log)

                    if is_success:
                        successes += 1

                    # Update progress bar
                    progress_bar.update(1)

                    # Save intermediate results if enabled
                    if (
                        config.get("save_temp_files", False)
                        and len(conversation_logs) % 10 == 0
                    ):
                        save_intermediate_results(
                            config, conversation_logs, successes, len(conversation_logs)
                        )
                except Exception as e:
                    prompt = future_to_prompt[future]
                    log(f"Error processing prompt '{prompt}': {e}", "error")

    # Calculate success rate
    success_rate = successes / total if total > 0 else 0
    end_time = time.time()
    total_time = end_time - start_time

    # Display results
    log("\nEXECUTION SUMMARY", "result")
    log(f"Success rate: {successes}/{total} ({success_rate:.2f}%)", "result")
    log(f"Total execution time: {total_time:.2f} seconds", "result")

    # Save patterns if using pattern memory
    if pattern_memory:
        if pattern_memory.analyze_logs(conversation_logs):
            log("Updated pattern memory with successful patterns", "success")

    return conversation_logs, success_rate


def save_intermediate_results(config, logs, successes, count):
    """
    Save intermediate results during long runs.

    Args:
        config (dict): Configuration dictionary
        logs (list): Conversation logs so far
        successes (int): Number of successes so far
        count (int): Number of prompts processed so far
    """
    # Create logs directory if it doesn't exist
    logs_dir = ensure_directory_exists(
        config.get("logs_directory", DEFAULT_PATHS["logs_directory"])
    )

    # Create temp file name
    temp_file = os.path.join(logs_dir, f"temp_results_{count}.csv")

    # Create run info
    run_info = {
        "Attacker Temperature": config["attacker_temp"],
        "Target Temperature": config["target_temp"],
        "Number of Turns": config["turns"],
        "Initial Attacker Rewrite Prompt": config["initial_prompt"][:50] + "...",
        "Followup Attacker Rewrite Prompt": (
            config["followup_prompt"][:50] + "..."
            if config["followup_prompt"]
            else "None"
        ),
        "StrongReject Threshold": config["strongreject_threshold"],
        "Target Model Key": config["target_model"],
        "Target Model": TARGET_MODELS[config["target_model"]]["name"],
        "Attacker Model": config["attacker_model"],
    }

    # Save logs
    success_rate_str = f"{(successes / count * 100):.2f}%" if count > 0 else "0.00%"
    save_conversation_log(run_info, logs, success_rate_str, temp_file)
    log(f"Saved intermediate results to {temp_file}", "info")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Turn Prompting Framework")

    # Model parameters
    parser.add_argument(
        "--target-model",
        type=str,
        default=DEFAULT_CONFIG["target_model"],
        help=f"Target model key. Options: {', '.join(TARGET_MODELS.keys())}",
    )
    parser.add_argument(
        "--target-temp",
        type=float,
        default=DEFAULT_CONFIG["target_temp"],
        help="Temperature for target model",
    )
    parser.add_argument(
        "--attacker-temp",
        type=float,
        default=DEFAULT_CONFIG["attacker_temp"],
        help="Temperature for attacker model",
    )
    parser.add_argument(
        "--attacker-model",
        type=str,
        default=DEFAULT_CONFIG["attacker_model"],
        help="Attacker model to use",
    )

    # Conversation parameters
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_CONFIG["turns"],
        help="Maximum number of conversation turns",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIG["strongreject_threshold"],
        help="StrongReject threshold for success",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        default=DEFAULT_CONFIG["target_memory_enabled"],
        help="Enable conversation memory for target model",
    )

    # Execution parameters
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_CONFIG["sample_size"],
        help="Number of prompts to sample (None for all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG["max_workers"],
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=DEFAULT_CONFIG["verbosity_level"],
        choices=[VERBOSE_NONE, VERBOSE_NORMAL, VERBOSE_DETAILED],
        help="Verbosity level",
    )

    # File paths
    parser.add_argument(
        "--prompts",
        type=str,
        default=DEFAULT_PATHS["adversarial_prompts"],
        help="Path to adversarial prompts CSV",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../Files/system_prompt.md"),
        help="Path to system prompt file",
    )
    parser.add_argument(
        "--followup-prompt",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../Files/system_prompt_followup.md"
        ),
        help="Path to followup system prompt file",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=DEFAULT_PATHS["logs_directory"],
        help="Directory for logs",
    )

    # Feature flags
    parser.add_argument(
        "--save-temp", action="store_true", help="Save intermediate results"
    )
    parser.add_argument(
        "--no-patterns", action="store_true", help="Disable pattern memory"
    )

    args = parser.parse_args()

    # Set global verbosity level
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = args.verbose

    # Check target model availability
    log(f"Checking if model '{args.target_model}' is available...", "info")
    if not is_model_available(args.target_model):
        log(
            f"Model '{args.target_model}' is not available. Please choose a different model.",
            "error",
        )
        return False

    # Load system prompts
    initial_prompt, followup_prompt = load_system_prompts(
        args.system_prompt, args.followup_prompt
    )
    if not initial_prompt:
        log("Failed to load system prompt.", "error")
        return False

    # Create configuration
    config = {
        "target_model": args.target_model,
        "target_model_name": TARGET_MODELS[args.target_model]["name"],
        "target_request_cost": TARGET_MODELS[args.target_model]["request_cost"],
        "target_temp": args.target_temp,
        "attacker_temp": args.attacker_temp,
        "attacker_model": args.attacker_model,
        "turns": args.turns,
        "strongreject_threshold": args.threshold,
        "target_memory_enabled": args.memory,
        "sample_size": args.sample_size,
        "max_workers": args.workers,
        "verbosity_level": args.verbose,
        "verbosity_level_name": VERBOSE_LEVEL_NAMES[args.verbose],
        "adversarial_prompts": args.prompts,
        "system_prompt": args.system_prompt,
        "system_prompt_followup": args.followup_prompt,
        "logs_directory": args.logs_dir,
        "save_temp_files": args.save_temp,
        "use_pattern_memory": not args.no_patterns,
        "initial_prompt": initial_prompt,
        "followup_prompt": followup_prompt,
    }

    # Display configuration
    display_config(config)

    # Run experiment
    conversation_logs, success_rate = run_experiment(config)

    # Save results
    if conversation_logs:
        # Create logs directory if it doesn't exist
        logs_dir = ensure_directory_exists(config["logs_directory"])

        # Get next available filename
        output_file = get_next_filename(logs_dir, "conversationLog")

        # Create run info
        run_info = {
            "Attacker Temperature": config["attacker_temp"],
            "Target Temperature": config["target_temp"],
            "Number of Turns": config["turns"],
            "Initial Attacker Rewrite Prompt": config["initial_prompt"],
            "Followup Attacker Rewrite Prompt": config["followup_prompt"],
            "StrongReject Threshold": config["strongreject_threshold"],
            "Target Model Key": config["target_model"],
            "Target Model": TARGET_MODELS[config["target_model"]]["name"],
            "Attacker Model": config["attacker_model"],
        }

        # Save logs
        success_rate_str = f"{(success_rate * 100):.2f}%"
        save_conversation_log(
            run_info, conversation_logs, success_rate_str, output_file
        )
        log(f"All conversation logs saved to {output_file}", "success")

    return True


if __name__ == "__main__":
    # Clear screen for fresh start
    os.system("cls" if os.name == "nt" else "clear")

    # Run the main function
    main()
