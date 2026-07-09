import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Model


class TransformersModel(Model):

    def __init__(
        self,
        model_name,
        device_map="auto",
        dtype="auto",
    ):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
        )

        self.model.eval()

    def generate(
        self,
        prompts,
        n=1,
        temperature=1.0,
        max_tokens=256,
    ):
        single_prompt = isinstance(prompts, str)

        if single_prompt:
            prompts = [prompts]

        if temperature <= 0 and n > 1:
            raise ValueError(
                "n > 1 requires sampling with temperature > 0"
            )

        texts = [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        do_sample = temperature > 0

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "num_return_sequences": n,
            "do_sample": do_sample,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        prompt_length = inputs["input_ids"].shape[1]

        completion_tokens = outputs[:, prompt_length:]

        completions = self.tokenizer.batch_decode(
            completion_tokens,
            skip_special_tokens=True,
        )

        if single_prompt and n == 1:
            return completions[0]

        return completions
