from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class LocalLLM:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "EleutherAI/gpt-neo-1.3B"

        print(f"Loading model {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.max_length = 1024
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def answer(self, prompt):
        max_len = self.model.config.max_position_embeddings
        max_new_tokens = 100

        max_input_length = max_len - max_new_tokens

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return answer
