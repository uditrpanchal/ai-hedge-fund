import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.ollama import ensure_ollama_and_model
import os # Added for os.getenv

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.utils.visualize import save_graph_as_png
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow if analysts are customized
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            agent = app

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash position. Defaults to 100000.0)")
    parser.add_argument("--margin-requirement", type=float, default=0.0, help="Initial margin requirement. Defaults to 0.0")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock ticker symbols")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")
    parser.add_argument("--default-analysts", type=str, help="Comma-separated list of default analyst keys to use in non-interactive mode (e.g., aswath_damodaran,ben_graham)")

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # Select analysts
    selected_analysts = None
    # If tickers are provided via CLI, assume non-interactive mode for analyst and model selection
    if args.tickers:
        print("Tickers provided via CLI. Detecting available LLM provider for non-interactive run...")
        
        # --- Analyst Selection Logic ---
        if args.default_analysts:
            # Ensure ANALYST_CONFIG keys are accessible for validation if needed
            # Or parse from ANALYST_ORDER's values
            all_available_analyst_keys = [key for _, key in ANALYST_ORDER]
            raw_default_analysts = [analyst.strip() for analyst in args.default_analysts.split(',')]
            selected_analysts = []
            valid_analysts_selected_display = [] # For storing display names
            invalid_analysts = []
            for analyst_key in raw_default_analysts:
                if analyst_key in all_available_analyst_keys:
                    selected_analysts.append(analyst_key)
                    # Find display name for printing
                    display_name = next((d_name for d_name, val in ANALYST_ORDER if val == analyst_key), analyst_key)
                    valid_analysts_selected_display.append(display_name.title().replace('_', ' '))
                else:
                    invalid_analysts.append(analyst_key)
            
            if invalid_analysts:
                print(f"{Fore.YELLOW}Warning: The following specified default analysts are invalid and will be ignored: {', '.join(invalid_analysts)}{Style.RESET_ALL}")
            if not selected_analysts: # If all provided defaults were invalid or none were provided that were valid
                 print(f"{Fore.YELLOW}No valid default analysts selected. No analysts will be run.{Style.RESET_ALL}")
                 # selected_analysts is already []
            else:
                 print(f"Using specified default analysts: {', '.join(Fore.GREEN + name + Style.RESET_ALL for name in valid_analysts_selected_display)}")

        else:
            selected_analysts = [] # Default to no analysts if --default-analysts is not provided
            print("No default analysts specified. No analysts will be run. Use --default-analysts to specify them.")
        # --- End of Analyst Selection Logic ---
        
        # MODIFIED SECTION STARTS HERE
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if deepseek_api_key: # Check if the key exists and is not empty
            model_provider = ModelProvider.DEEPSEEK.value
            # Select a default DeepSeek model
            default_deepseek_model_info = next((m for m in LLM_ORDER if m[2] == ModelProvider.DEEPSEEK.value and not m[0].endswith("(Custom)")), None)
            if default_deepseek_model_info:
                model_name = default_deepseek_model_info[1] # e.g., 'deepseek-chat'
                print(f"Using DeepSeek model: {model_name} as DEEPSEEK_API_KEY is available in .env.")
            else:
                # Fallback if no default is found in LLM_ORDER, though this shouldn't happen with proper config
                model_name = "deepseek-chat" # A common default
                print(f"{Fore.YELLOW}No default DeepSeek model found in LLM_ORDER, defaulting to '{model_name}'. Ensure LLM_ORDER is configured.{Style.RESET_ALL}")
                print(f"Using DeepSeek model: {model_name} as DEEPSEEK_API_KEY is available in .env.")
        else:
            print(f"{Fore.RED}DEEPSEEK_API_KEY not found or is empty in .env file. This is required for non-interactive mode.{Style.RESET_ALL}")
            sys.exit(1)
        # MODIFIED SECTION ENDS HERE
            
        # The print statement for analysts is now handled within the selection logic above.
        # If selected_analysts is empty, this would print a confusing message or error.
        # if selected_analysts: 
        #    print(f"\nUsing default analysts: {', '.join(Fore.GREEN + sa.title().replace('_', ' ') + Style.RESET_ALL for sa in selected_analysts)}")
        print(f"Using default model: {Fore.CYAN}{model_provider}{Style.RESET_ALL} - {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
    else: # Interactive mode
        choices = questionary.checkbox(
            "Select your AI analysts.",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
            validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
            style=questionary.Style(
                [
                    ("checkbox-selected", "fg:green"),
                    ("selected", "fg:green noinherit"),
                    ("highlighted", "noinherit"),
                    ("pointer", "noinherit"),
                ]
            ),
        ).ask()

        if not choices:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        else:
            selected_analysts = choices
            print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

        # Select LLM model based on whether Ollama is being used
        model_name = ""
        model_provider = ""

        if args.ollama:
            print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")

            # Select from Ollama-specific models
            model_name: str = questionary.select(
                "Select your Ollama model:",
                choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
                style=questionary.Style(
                    [
                        ("selected", "fg:green bold"),
                        ("pointer", "fg:green bold"),
                        ("highlighted", "fg:green"),
                        ("answer", "fg:green bold"),
                    ]
                ),
            ).ask()

            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

            if model_name == "-":
                model_name = questionary.text("Enter the custom model name:").ask()
                if not model_name:
                    print("\n\nInterrupt received. Exiting...")
                    sys.exit(0)

            # Ensure Ollama is installed, running, and the model is available
            if not ensure_ollama_and_model(model_name):
                print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
                sys.exit(1)

            model_provider = ModelProvider.OLLAMA.value
            print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        else:
            # Use the standard cloud-based LLM selection
            model_choice = questionary.select(
                "Select your LLM model:",
                choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
                style=questionary.Style(
                    [
                        ("selected", "fg:green bold"),
                        ("pointer", "fg:green bold"),
                        ("highlighted", "fg:green"),
                        ("answer", "fg:green bold"),
                    ]
                ),
            ).ask()

            if not model_choice:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

            model_name, model_provider = model_choice

            # Get model info using the helper function
            model_info = get_model_info(model_name, model_provider)
            if model_info:
                if model_info.is_custom():
                    model_name = questionary.text("Enter the custom model name:").ask()
                    if not model_name:
                        print("\n\nInterrupt received. Exiting...")
                        sys.exit(0)

                print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
            else:
                model_provider = "Unknown" # Should not happen if choice is from LLM_ORDER
                print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    # Create the workflow with selected analysts
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = ""
        if selected_analysts is not None:
            for selected_analyst in selected_analysts:
                file_path += selected_analyst + "_"
            file_path += "graph.png"
        save_graph_as_png(app, file_path)

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Initialize portfolio with cash amount and stock positions
    portfolio = {
        "cash": args.initial_cash,  # Initial cash amount
        "margin_requirement": args.margin_requirement,  # Initial margin requirement
        "margin_used": 0.0,  # total margin usage across all short positions
        "positions": {
            ticker: {
                "long": 0,  # Number of shares held long
                "short": 0,  # Number of shares held short
                "long_cost_basis": 0.0,  # Average cost basis for long positions
                "short_cost_basis": 0.0,  # Average price at which shares were sold short
                "short_margin_used": 0.0,  # Dollars of margin used for this ticker's short
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # Realized gains from long positions
                "short": 0.0,  # Realized gains from short positions
            }
            for ticker in tickers
        },
    }

    # Run the hedge fund
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )
    print_trading_output(result)
