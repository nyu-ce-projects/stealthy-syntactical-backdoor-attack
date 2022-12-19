import math
import torch
import numpy as np
import transformers
import logging
import os
        
        
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

class GPT2LM:
    def __init__(self) -> None:

        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", from_tf=False)

    def __call__(self, sent):
        ipt = self.tokenizer(sent, return_tensors="pt", verbose=False,  )
        lm = self.lm(input_ids=ipt['input_ids'],
                                attention_mask=ipt['attention_mask'],
                                labels=ipt.input_ids)
        try:
            ppl = math.exp(lm[0])
        except RuntimeError:
            ppl = np.nan
        return ppl,lm


