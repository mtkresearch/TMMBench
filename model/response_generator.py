from typing import List, Union
from datasets import load_dataset, Dataset
from pathlib import Path
import json
import os
import argparse

from model.backend import (
    openai_api,
    vllm_backend,
    hf_backend
)

BACKENDS = {
    "vllm": vllm_backend,
    "hf": hf_backend,
    "openai_api": openai_api,
}


class ResponseGenerator:
    """
    A class to load questions, perform inference using specified model backends,
    and save generated responses to a JSONL file.
    """

    def __init__(
        self,
        model_name: str,
        log_dir: str,
        question_path: str = "eval_data/Question_multiplechoice.tsv",
        backend: str = "hf",
        selected_category: Union[str, List[str]] = "ALL"
    ):
        """
        Initialize the ResponseGenerator.

        Args:
            model_name (str): Name or path of the model.
            log_dir (str): Directory to store logs (e.g., JSONL output).
            question_path (str): Path to the TSV file containing questions.
            backend (str): Name of the inference backend (e.g., "hf", "vllm").
            selected_category (str, List[str]): Category or categories to filter.
                If "ALL", no filtering is applied.
        """
        self.model_name = model_name
        self.log_dir = log_dir
        self.question_path = question_path
        self.backend = backend
        self.selected_category = selected_category

        # Additional parameters
        self.temperature = 0.01

        # Load the dataset and add meta category
        self.questions_dataset = self.load_questions(self.question_path)
        self.questions_dataset = self.questions_dataset.map(self.add_meta_category)

        # Prepare model with the specified backend
        self.model = self._prepare_inference_backend(self.backend)

    def load_questions(self, question_path: str) -> Dataset:
        """
        Load the TSV or parquet dataset of questions and optionally filter by category.

        Args:
            question_path (str): Path to the dataset.

        Returns:
            Dataset: A Hugging Face Dataset object containing questions.
        """
        if question_path.endswith(".tsv"):
            dataset = load_dataset("csv", data_files=question_path, delimiter="\t")[
                "train"
            ]
        elif question_path.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=question_path)["train"]

        else:
            raise ValueError("Unsupported file format. Please use .tsv or .parquet.")

        # Filter by category if it is not "ALL"
        if self.selected_category != "ALL":
            # If selected_category is a single string, wrap it in a list:
            categories = (
                [self.selected_category]
                if isinstance(self.selected_category, str)
                else self.selected_category
            )
            dataset = dataset.filter(
                lambda example: example["category"] in categories
            )

        return dataset

    def add_meta_category(self, example: dict) -> dict:
        """
        Add a meta_category field based on the existing category field.

        Args:
            example (dict): A single dataset example with "category".

        Returns:
            dict: The example with an added "meta_category" field.
        """
        cat2meta = {
            "civil": "social_science",
            "UI_understanding": "UI_understanding",
            "celebrity": "celebrity",
            "diagram": "diagram",
            "news": "daily_life",
            "geography": "social_science",
            "physics": "STEM",
            "attractions": "attractions",
            "taiwan_road_sign": "daily_life",
            "biology": "STEM",
            "infographics": "infographics",
            "taiwan_ad": "daily_life",
            "history": "social_science",
            "chemistry": "STEM",
            "math": "STEM",
            "table": "table",
        }
        # Use .get() to avoid KeyError for unknown categories
        example["meta_category"] = cat2meta.get(example["category"], "others")
        return example

    def _prepare_inference_backend(self, backend: str):
        """
        Prepare the appropriate inference backend based on user selection.

        Args:
            backend (str): Name of the inference backend.

        Returns:
            An instance of the backend class.

        Raises:
            NotImplementedError: If the backend is unknown.
        """

        # For all others, if not in BACKENDS, raise exception
        if backend not in BACKENDS:
            raise NotImplementedError(f"Backend '{backend}' is not supported.")

        return BACKENDS[backend](self.model_name, self.temperature)

    def generate_responses(self) -> List[str]:
        """
        Generate model responses for the dataset loaded in self.questions_dataset.

        Returns:
            List[str]: The generated responses from the model.
        """
        # The .inference() method is assumed to be implemented by each backend
        model_response = self.model.inference(self.questions_dataset)

        # Basic consistency check
        assert len(model_response) == len(self.questions_dataset), (
            "Mismatch between the number of responses and the dataset length."
        )

        # Add the model's response as a new column in the dataset
        self.questions_dataset = self.questions_dataset.add_column(
            "response", model_response
        )

        # Save responses to a JSONL file
        self.save_to_jsonl()
        print("Finished generating model responses!")
        return model_response

    def save_to_jsonl(self) -> None:
        """
        Save the dataset with model responses to a JSONL file in self.log_dir.
        """
        # Columns to be saved
        columns_to_write = [
            "id",
            "meta_category",
            "category",
            "question",
            "answer",
            "A",
            "B",
            "C",
            "D",
            "E",
            "response",
        ]

        # Create the log directory if it does not exist
        output_dir = Path(self.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = os.path.join(output_dir, "response.jsonl")
        with open(save_path, "w", encoding="utf-8") as f:
            for row in self.questions_dataset:
                selected_columns = {col: row[col] for col in columns_to_write if col in row}
                f.write(json.dumps(selected_columns, ensure_ascii=False) + "\n")

        print(f"Responses have been written to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Response Generator Script")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the model"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory to store logs"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        help="Inference backend (e.g., 'hf', 'vllm', 'openai_api', etc.)"
    )
    parser.add_argument(
        "--selected_category",
        type=str,
        default="ALL",
        help="Category or categories to filter. If 'ALL', no filtering."
    )

    args = parser.parse_args()

    response_generator = ResponseGenerator(
        model_name=args.model_name,
        log_dir=args.log_dir,
        question_path="eval_data/Question_multiplechoice.tsv",
        backend=args.backend,
        selected_category=args.selected_category
    )

    response_generator.generate_responses()