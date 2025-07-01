import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LocalLLM:
    """
    GPT-Neo 1.3 B wrapper that:
    """

    def __init__(self, model_name: str = "EleutherAI/gpt-neo-1.3B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # fix padding warning
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.max_ctx = self.model.config.max_position_embeddings  # 2048

    def answer(self, prompt: str, max_new_tokens: int = 120) -> str:
        max_input = self.max_ctx - max_new_tokens
        toks = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
            padding=True,
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}

        out = self.model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_only = out[0][toks["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

