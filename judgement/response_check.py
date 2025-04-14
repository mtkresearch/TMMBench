from typing import List, Dict, Optional, Any, Union
import json
from datasets import Dataset, load_dataset
from pathlib import Path
import re

from model.backend import openai_api


class ResponseChecker:
    """
    Class for evaluating and scoring model responses for either MCQ or free-form questions.
    """

    def __init__(self, response_file: str, log_dir: str) -> None:
        """
        Initialize the ResponseChecker.

        Args:
            response_file (str): Path to the JSONL file containing model responses.
            log_dir (str): Directory where log files will be written.
        """
        self.response_data: Dataset = self._load_response(response_file)
        self.judgements_result: Optional[List[bool]] = None
        self.log_dir: str = log_dir

    def _load_response(self, response_file: str) -> Dataset:
        """
        Load the evaluated model's JSONL response file into a HuggingFace Dataset.

        Args:
            response_file (str): Path to the JSONL response file.

        Returns:
            Dataset: A HuggingFace Dataset containing the response data.
        """
        return load_dataset("json", data_files=response_file, split="train")

    def generate_judgments_api(self) -> None:
        """
        Call the AI model's API to generate a list of judgments for each response.
        Then parse and store these judgments as boolean correctness values in self.judgements_result.
        """
        model = openai_api(model_identifier="gpt-4o", temperature=0.01)
        judgment_list = model.inference(self.response_data, mode="judgment")

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{self.log_dir}/judgment_list.json", "w", encoding="utf-8") as f:
            json.dump(judgment_list, f, indent=4, ensure_ascii=False)

        parsed_judgment_list = [self.parser(j) for j in judgment_list]

        with open(f"{self.log_dir}/parsed_judgment_list.json", "w", encoding="utf-8") as f:
            json.dump(parsed_judgment_list, f, indent=4, ensure_ascii=False)

        self.judgements_result = [
            self.response_data["answer"][i] == parsed_judgment_list[i]
            for i in range(len(parsed_judgment_list))
        ]

    def parser(self, input_string: str) -> Optional[str]:
        """
        Extract the assistant's choice from the provided string using regex.

        Args:
            input_string (str): The string containing an "Assistant choice:" line.

        Returns:
            Optional[str]: The extracted choice, or None if the pattern is not found.
        """
        pattern = r"Assistant choice:\s*(.+)$"
        match = re.search(pattern, input_string)
        if match:
            return match.group(1).strip()
        return None

    def calculate_score(self) -> None:
        """
        Use the judgments (self.judgements_result) to calculate category-level
        and overall accuracy. Outputs a JSON file with calculated scores.
        This method will also write the combined results to a log file.
        """
        if self.judgements_result is None:
            raise ValueError("No judgments found. Run generate_judgments_api() first.")

        # Assign the judgment results to the dataset
        def assign_result(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
            example["is_correct"] = self.judgements_result[idx]
            return example

        data = self.response_data.map(assign_result, with_indices=True)

        # Build a mapping from meta_category -> (set of categories)
        meta_category_to_categories: Dict[str, set] = {}
        for row in data:
            meta_cat = row["meta_category"]
            cat = row["category"]
            if meta_cat not in meta_category_to_categories:
                meta_category_to_categories[meta_cat] = set()
            meta_category_to_categories[meta_cat].add(cat)

        # Single-pass calculation of per-category counts
        category_counts: Dict[str, int] = {}
        category_correct_counts: Dict[str, int] = {}
        total_correct_answers: int = 0
        total_questions_count: int = 0

        for row in data:
            cat = row["category"]
            if cat not in category_counts:
                category_counts[cat] = 0
                category_correct_counts[cat] = 0

            category_counts[cat] += 1
            if row["is_correct"]:
                category_correct_counts[cat] += 1

        # Compute accuracy per category
        category_accuracy: Dict[str, float] = {}
        for cat, count in category_counts.items():
            correct = category_correct_counts[cat]
            accuracy = (correct / count) * 100 if count > 0 else 0
            category_accuracy[cat] = accuracy
            total_correct_answers += correct
            total_questions_count += count

        # Compute overall accuracy
        overall_accuracy = (
            (total_correct_answers / total_questions_count) * 100
            if total_questions_count > 0
            else 0
        )
        category_accuracy["Overall"] = overall_accuracy

        # Build detailed accuracy table
        accuracy_table = [{"overall_accuracy": f"{overall_accuracy:.4f}%"}]

        # Compute meta-category accuracy
        for meta_cat, cat_set in meta_category_to_categories.items():
            meta_total_correct = 0
            meta_total_count = 0
            # Aggregate stats from categories belonging to meta_cat
            for c in cat_set:
                meta_total_correct += (category_accuracy[c] / 100) * category_counts[c]
                meta_total_count += category_counts[c]
            meta_accuracy = (
                (meta_total_correct / meta_total_count) * 100
                if meta_total_count > 0
                else 0
            )

            # Add meta-category details to the table
            if len(cat_set) > 1:
                # If there are multiple sub-categories, show each one
                accuracy_table.append(
                    {
                        "meta_category": meta_cat,
                        "meta_category_accuracy": f"{meta_accuracy:.4f}%",
                        "sub_category": {
                            c: f"{category_accuracy[c]:.4f}%"
                            for c in cat_set
                        },
                    }
                )
            else:
                accuracy_table.append(
                    {
                        "meta_category": meta_cat,
                        "meta_category_accuracy": f"{meta_accuracy:.4f}%",
                    }
                )

        # Write the accuracy table to file
        self.write_to_file(accuracy_table)

    def write_to_file(self, data: Union[Dict[str, Any], List[Any]]) -> None:
        """
        Write the judgments results and scores to a JSON file in log_dir.

        Args:
            data (Union[Dict[str, Any], List[Any]]): Data to be written to JSON.
        """
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{self.log_dir}/category_score.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
