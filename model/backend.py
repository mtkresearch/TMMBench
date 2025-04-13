from typing import List, Dict, Any
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import AutoModel, AutoTokenizer, AutoProcessor
from openai import OpenAI, APIError
import os
from transformers import GenerationConfig

from mtkresearch.llm.prompt import MRPromptV3


class inference_backend(ABC):
    """
    Abstract base class for inference backends
    """
    
    @abstractmethod
    def __init__(self, model_identifier: str, temperature: float, max_tokens: int):
        self.model_identifier = model_identifier
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights and initialize processor"""
        
    @abstractmethod
    def inference(self, dataset: List[Dict[str, Any]], **kwargs) -> List[str]:
        """Process dataset and return responses"""

    @abstractmethod
    def _construct_inputs(self, example: Dict[str, Any]) -> Any:
        """Construct model-specific inputs from example"""


class vllm_backend(inference_backend):
    def __init__(self, model_identifier: str, temperature: float = 0.01, max_tokens: int = 2048):
        super().__init__(model_identifier, temperature, max_tokens)
        from vllm import SamplingParams
        self.processor = AutoProcessor.from_pretrained(model_identifier)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def _load_model(self) -> None:
        from vllm import LLM
        self.model = LLM(
            model=self.model_identifier,
            trust_remote_code=True,
            tensor_parallel_size=8,
            gpu_memory_utilization=0.8,
            max_num_seqs=16,
            max_model_len=8192 * 2,
            enforce_eager=True
        )

    def inference(self, dataset: List[Dict[str, Any]], **kwargs) -> List[str]:
        inputs_list = [self._construct_inputs(example) for example in dataset]
        return [
            self.model.generate(inputs, self.sampling_params)[0].outputs[0].text
            for inputs in inputs_list
        ]

    def _construct_inputs(self, example: Dict[str, Any]) -> Any:
        prompt = f"{example['question']}\nA:{example['A']}\nB:{example['B']}\nC:{example['C']}\nD:{example['D']}"
        if example["E"] is not None:
            prompt += f"\nE:{example['E']}"
        
        messages = [{"role": "user", "content": [
            {"type": "image"}, 
            {"type": "text", "text": prompt}
        ]}]
        
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        image = Image.open(f"data/image/mcq_image_{example['id']}.png")
        return {
            "prompt": input_text,
            "multi_modal_data": {"image": image}
        }


class hf_backend(inference_backend):
    def __init__(self, model_identifier: str, temperature: float = 0.01, max_tokens: int = 4096):
        super().__init__(model_identifier, temperature, max_tokens)
        # TODO: support for other VLM other than Breeze2. Note that the methods in this class is for Breeze2 only.
        self.prompt_engine = MRPromptV3()

    def _load_model(self) -> None:
        self.model = AutoModel.from_pretrained(
            self.model_identifier,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            img_context_token_id=128212
        ).eval().cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_identifier, 
            trust_remote_code=True, 
            use_fast=False
        )

    def inference(self, dataset: List[Dict[str, Any]], **kwargs) -> List[str]:
        input_list = [self._construct_inputs(example) for example in dataset]
        generation_config = GenerationConfig(
            max_new_tokens=self.max_tokens,
            do_sample=False,
            eos_token_id=128009
        )
        
        return [
            self.prompt_engine.parse_generated_str(
                self._inference(generation_config, prompt, pixel_values)
            )["content"]
            for prompt, pixel_values in tqdm(input_list)
        ]

    def _construct_inputs(self, example: Dict[str, Any]) -> Any:
        prompt = f"{example['question']}\nA:{example['A']}\nB:{example['B']}\nC:{example['C']}\nD:{example['D']}"
        if example["E"] is not None:
            prompt += f"\nE:{example['E']}"
        
        image_path = f"data/image/mcq_image_{example['id']}.png"
        conversations = [{
            "role": "user",
            "content": [
                {"type": "image", "image_path": image_path},
                {"type": "text", "text": prompt}
            ]
        }]
        
        return self.prompt_engine.get_prompt(conversations)

    def _inference(self, generation_config: GenerationConfig, prompt: str, pixel_values: Any) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        pixel_values = pixel_values.to(self.model.dtype).to(self.model.device)
        output_tensors = self.model.generate(
            **inputs,
            generation_config=generation_config,
            pixel_values=pixel_values
            
        )
        return self.tokenizer.decode(output_tensors[0])


class openai_api(inference_backend):
    def __init__(self, model_identifier: str, temperature: float = 0.01, max_tokens: int = 4096):
        super().__init__(model_identifier, temperature, max_tokens)
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=None,
            timeout=20.0,
            max_retries=10
        )

    def _load_model(self) -> None:
        if self.model_identifier not in ["gpt-4o", "gpt-4o-mini"]:
            raise ValueError("Unsupported model name")

    def inference(self, dataset: List[Dict[str, Any]], mode: str = "response", **kwargs) -> List[str]:
        message_constructor = {
            "response": self._construct_messages,
            "judgment": self._construct_judgment_messages
        }.get(mode)
        
        if not message_constructor:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        return [
            self._query_openai(message_constructor(example))
            for example in tqdm(dataset, desc="Processing OpenAI API calls")
        ]

    def _construct_inputs(self, example: Dict[str, Any]) -> Any:
        return self._construct_messages(example)

    def _query_openai(self, messages: List[Dict[str, Any]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_identifier,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except APIError as e:
            print(f"OpenAI API error: {e}")
            return ""

    def _construct_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = f"{example['question']}\nA:{example['A']}\nB:{example['B']}\nC:{example['C']}\nD:{example['D']}"
        if example["E"] is not None:
            prompt += f"\nE:{example['E']}"

        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{example['image']}"}
                    }
                ]
            }
        ]

    def _construct_judgment_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_template = """The following is a response from an assistant given a multiple choice question with only one correct answer.
You have to determine which choice the assistant selected.
All the choices are:
A: {A}, B: {B}, C: {C}, D: {D}{E_choice}.
The response from the assistant is:
{RESPONSE}.
After evaluating the response, you need to:
- Briefly state your judgment in up to 100 words.
- Conclude the assistant's choice in the format below. If the assistant does not respond or selects multiple choices, respond with NA in the field <the choice>.
The field <the choice> should be an English alphabet.
"Assistant choice: <the choice>"
"""
        E_choice = f", E: {example['E']}" if example["E"] is not None else ""
        prompt = prompt_template.format(
            A=example["A"],
            B=example["B"],
            C=example["C"],
            D=example["D"],
            E_choice=E_choice,
            RESPONSE=example["response"],
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        return messages
