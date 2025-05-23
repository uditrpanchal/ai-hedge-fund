from typing_extensions import Annotated, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage
from colorama import Fore, Style # Added for colored output

import json


def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    return {**a, **b}


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]


def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    # Prioritize and print reasoning_text if available
    if isinstance(output, dict):
        # Check for top-level reasoning_text (e.g. if the whole agent had one message)
        if "reasoning_text" in output and output["reasoning_text"]:
            print(f"{Fore.YELLOW}Note: {output['reasoning_text']}{Style.RESET_ALL}")
            # If top-level reasoning_text is present, we might consider if we still need to print the full JSON.
            # For now, we'll print it, but this could be a point of future refinement.

        # For agents like Technical Analyst, output is a dict per ticker.
        # Check each ticker's analysis for 'reasoning_text'.
        # This loop assumes output is a dictionary where values might be dictionaries containing 'reasoning_text'.
        # It's designed to be general for various output structures.
        for key, value_data in output.items(): # 'key' could be ticker, or any other primary key in the output
            if isinstance(value_data, dict) and "reasoning_text" in value_data and value_data["reasoning_text"]:
                print(f"{Fore.CYAN}Note for [{key.upper()}]: {Style.RESET_ALL}{Fore.YELLOW}{value_data['reasoning_text']}{Style.RESET_ALL}")

    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)): # Added bool
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    if isinstance(output, (dict, list)):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            print(output)

    print("=" * 48)
