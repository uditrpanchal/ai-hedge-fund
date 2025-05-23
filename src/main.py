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
    parser.add_argument("--llm-provider", type=str, choices=[provider.value for provider in ModelProvider], help="Specify the LLM provider non-interactively (e.g., deepseek, ollama).")
    parser.add_argument("--llm-model", type=str, help="Specify the LLM model name non-interactively.")

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    # Note: args.tickers is required by argparse, so it will always be present.
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # --- Analyst Selection ---
    selected_analysts = None
    if args.default_analysts:
        # Non-interactive analyst selection logic
        print("Using default analysts specified via --default-analysts.")
        all_available_analyst_keys = [key for _, key in ANALYST_ORDER]
        raw_default_analysts = [analyst.strip() for analyst in args.default_analysts.split(',')]
        selected_analysts = []
        valid_analysts_selected_display = []
        invalid_analysts = []
        for analyst_key in raw_default_analysts:
            if analyst_key in all_available_analyst_keys:
                selected_analysts.append(analyst_key)
                display_name = next((d_name for d_name, val in ANALYST_ORDER if val == analyst_key), analyst_key)
                valid_analysts_selected_display.append(display_name.title().replace('_', ' '))
            else:
                invalid_analysts.append(analyst_key)
        
        if invalid_analysts:
            print(f"{Fore.YELLOW}Warning: The following specified default analysts are invalid and will be ignored: {', '.join(invalid_analysts)}{Style.RESET_ALL}")
        if not selected_analysts:
            print(f"{Fore.YELLOW}No valid default analysts selected. No analysts will be run.{Style.RESET_ALL}")
        else:
            print(f"Using specified default analysts: {', '.join(Fore.GREEN + name + Style.RESET_ALL for name in valid_analysts_selected_display)}\n")
    else:
        # Interactive analyst selection using questionary
        print("No default analysts specified, proceeding with interactive selection.")
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
            print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in selected_analysts)}\n")

    # --- LLM Model and Provider Selection ---
    model_name = ""
    model_provider = ""

    if args.llm_provider and args.llm_model:
        print("Using non-interactive LLM selection based on --llm-provider and --llm-model arguments.")
        model_provider = args.llm_provider
        model_name = args.llm_model

        if model_provider == ModelProvider.OLLAMA.value:
            if not ensure_ollama_and_model(model_name):
                print(f"{Fore.RED}Ollama setup for model '{model_name}' failed. Cannot proceed.{Style.RESET_ALL}")
                sys.exit(1)
            # print(f"Verified Ollama and model '{model_name}'.") # Optional: too verbose
        elif model_provider == ModelProvider.DEEPSEEK.value:
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                print(f"{Fore.RED}DEEPSEEK_API_KEY not found in .env file. This is required for the '{model_provider}' provider.{Style.RESET_ALL}")
                sys.exit(1)
            # print(f"DEEPSEEK_API_KEY found. Proceeding with provider '{model_provider}'.") # Optional: too verbose
        # Add more provider checks here if necessary in the future (e.g., for OPENAI_API_KEY if OpenAI is non-interactively selected)
        elif model_provider == ModelProvider.OPENAI.value: # Example for OpenAI non-interactive
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print(f"{Fore.RED}OPENAI_API_KEY not found in .env file. This is required for the '{model_provider}' provider.{Style.RESET_ALL}")
                sys.exit(1)

        print(f"\nUsing specified LLM provider: {Fore.CYAN}{model_provider}{Style.RESET_ALL} and model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    else: # Fallback to interactive LLM selection
        print("Proceeding with interactive LLM selection (as --llm-provider and --llm-model were not both specified).")
        if args.ollama:
            print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")
            model_name_str: str = questionary.select( 
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
            model_name = model_name_str 

            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

            if model_name == "-":
                model_name = questionary.text("Enter the custom model name:").ask()
                if not model_name:
                    print("\n\nInterrupt received. Exiting...")
                    sys.exit(0)

            if not ensure_ollama_and_model(model_name):
                print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
                sys.exit(1)

            model_provider = ModelProvider.OLLAMA.value
            print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        else:
            # Interactive standard cloud-based LLM selection
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
            model_info = get_model_info(model_name, model_provider) # model_info might be None if it's a custom model not in LLM_ORDER

            if model_info and model_info.is_custom(): # if model_info is None, it's treated as custom too.
                 model_name = questionary.text(f"Enter the custom model name for {model_provider}:").ask()
                 if not model_name:
                    print("\n\nInterrupt received. Exiting...")
                    sys.exit(0)
            elif not model_info and not (model_name == "-" or any(model_name == m[1] for m in LLM_ORDER)): # If not in LLM_ORDER and not a custom placeholder
                # This case is for when a model name was entered that's not in LLM_ORDER and not marked custom.
                # It's effectively a custom model name.
                # The current logic assumes custom models are explicitly marked with '-' or similar.
                # For safety, if a model is not in LLM_ORDER and not explicitly custom, we can treat it as such or warn.
                # For now, we assume model_name is final if it wasn't '-'
                 pass


            # API Key checks for interactively selected cloud providers
            if model_provider == ModelProvider.DEEPSEEK.value:
                deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
                if not deepseek_api_key:
                    print(f"{Fore.RED}DEEPSEEK_API_KEY not found in .env file. This is required for the selected '{model_provider}' provider.{Style.RESET_ALL}")
                    sys.exit(1)
            elif model_provider == ModelProvider.OPENAI.value: # Example for OpenAI interactive
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    print(f"{Fore.RED}OPENAI_API_KEY not found in .env file. This is required for the selected '{model_provider}' provider.{Style.RESET_ALL}")
                    sys.exit(1)

            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
            
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
