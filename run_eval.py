import argparse
import os

from judgement.response_check import ResponseChecker
from model.response_generator import ResponseGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Response Generator Script")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name or path of the model",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Directory to store logs",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        help="Inference backend (e.g., 'hf', 'vllm', 'openai_api', etc.)",
    )
    parser.add_argument(
        "--selected_category",
        type=str,
        default="ALL",
        help="Category or categories to filter. If 'ALL', no filtering.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    response_generator = ResponseGenerator(
        model_name=args.model_name,
        log_dir=args.log_dir,
        backend=args.backend,
        selected_category=args.selected_category,
    )
    response_generator.generate_responses()

    checker = ResponseChecker(
        response_file=os.path.join(args.log_dir, "response.jsonl"),
        log_dir=args.log_dir,
    )
    checker.generate_judgments_api()
    checker.calculate_score()


if __name__ == "__main__":
    main()
