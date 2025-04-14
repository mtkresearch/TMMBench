## TMMBench - Taiwan Multi-modal Model Benchmark

TMMBench is a collection of vision-language model benchmarks for traditional Chinese and Taiwan-specific topics.

The benchmark is composed of 9 categories, including 290 questions:

1. STEM (25 questions)

   - Taiwanese college entrance exam: Mathematics (6 questions)
   - Taiwanese college entrance exam: Chemistry (6 questions)
   - Taiwanese college entrance exam: Biology (6 questions)
   - Taiwanese college entrance exam: Physics (6 questions)
2. Humanities and Social Sciences (20 questions)

   - Taiwanese college entrance exam: Geography (7 questions)
   - Taiwanese college entrance exam: History (6 questions)
   - Taiwanese college entrance exam: Civics and Society (7 questions)
3. Tables (35 questions)
4. Infographics (35 questions)
5. Diagram (35 questions)
6. Daily Life in Taiwan (35 questions)

   - Road signs (6 questions)
   - Promotional advertisements (17 questions)
   - News (12 questions)
7. Celebrity (35 questions)
8. Attractions and Landmarks (35 questions)
9. UI understanding (35 questions)

## How to Use the Benchmark

### Prerequisites

1. **Install Dependencies**

   ```sh
   pip install -r requirements
   ```

2. **Download Benchmark Data**

   Download the evaluation data from the `MediaTek-Research/TMMBench` repository and set up as follows:
   - Create an `eval_data` folder in the project root if it doesn't exist
   - Place `Question_multiplechoice.tsv` in the `eval_data` folder

### Running the Benchmark

1. **Set Up OpenAI API Key**

   ```bash
   export OPENAI_API_KEY="your_openai_key"
   ```

2. **Execute the Evaluation Script**

   The `run_eval.py` script handles both response generation and judgment via GPT-4o:

   ```bash
   python run_eval.py
   ```

3. **Additional Options**

   You can customize the evaluation with these parameters:
   ```bash
   python run_eval.py --model_name MODEL_PATH --backend hf --log_dir logs/YOUR_MODEL/
   ```

   - `--model_name`: Path to your model (default: Llama-Breeze2-3B-Instruct)
   - `--backend`: Inference backend (options: hf, vllm, openai_api)
   - `--log_dir`: Directory to store results


## Citation 

```
@software{tmmbench,
  author = {Chia-Sheng Liu and Yi-Chang Chen and Yu-Ting Hsu and Ru-Hung Huang and Meng-Hsi Chen and Da-Shan Shiu},
  title = {TMMBench - Taiwan Multi-modal Model Benchmark},
  month = April,
  year = 2025,
  url = {https://github.com/mtkresearch/TMMBench}
}
```
