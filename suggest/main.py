from functools import partial
from typing import List

import torch
from fastapi import FastAPI
from jellyfish import damerau_levenshtein_distance
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelWithLMHead, top_k_top_p_filtering, pipeline
from torch.nn import functional as F

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

casual_tokenizer = AutoTokenizer.from_pretrained("gpt2")
casual_model = AutoModelWithLMHead.from_pretrained("gpt2")

mask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
mask_model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")


def suggestions_from_best_tokens(best_tokens, hint, top_k):
    if not hint:
        return best_tokens[:top_k]

    starts_with_hint, doesnt = [], []
    for x in best_tokens:
        starts_with_hint.append(x) if x.startswith(hint) else doesnt.append(x)

    if len(starts_with_hint) >= top_k:
        return starts_with_hint[:top_k]

    sort_by = partial(damerau_levenshtein_distance, hint)

    # TODO optimize
    most_similar = sorted(doesnt, key=sort_by)

    return starts_with_hint + most_similar[:top_k-len(starts_with_hint)]


@app.get("/suggestions/next")
def get_suggestions_next(text_before: str = "", hint: str = "", top_k: int = 3) -> List[str]:
    input_ids = casual_tokenizer.encode(text_before, return_tensors="pt")

    # get logits of last hidden state
    next_token_logits = casual_model(input_ids)[0][:, -1, :]

    # filter
    # filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

    # sample
    best_token_ids = torch.topk(next_token_logits, 5000, dim=1).indices[0].tolist()
    best_tokens = [casual_tokenizer.decode(t) for t in best_token_ids]

    return suggestions_from_best_tokens(best_tokens, hint, top_k)


@app.get("/suggestions/between")
def get_suggestions_between(text_before: str = "", hint: str = "", text_after: str = "", top_k: int = 3) -> List[str]:
    input = mask_tokenizer.encode(f"{text_before} {mask_tokenizer.mask_token} {text_after}", return_tensors="pt")
    mask_token_index = torch.where(input == mask_tokenizer.mask_token_id)[1]
    token_logits = mask_model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    best_token_ids = torch.topk(mask_token_logits, 5000, dim=1).indices[0].tolist()
    best_tokens = [mask_tokenizer.decode([t]) for t in best_token_ids]

    return suggestions_from_best_tokens(best_tokens, hint, top_k)
