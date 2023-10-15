from typing import Type
import html
import re

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# Unicode hex values
curly_lsquote = '\u2018'
curly_rsquote = '\u2019'
curly_ldquote = '\u201C'
curly_rdquote = '\u201D'
em_dash = '\u2014'
double_plus = '\u29FA'
ellipsis = '\u2026'
en_dash = '\u2013'
uml_I = "\u00CF"
delta = "\u2206"
delta2 = "\u25B3"
degree = "\u02DA"
one = "\u0661"
nine = "\u0669"
AA = "\u0041\u030A"
ue = "\u0075\u0308"
heart = "\u2661"
fill_heart = "\u2764"


# Define replacement dictionary for blacklists
blacklist_replace_dict = {
    curly_lsquote: "'",
    curly_rsquote: "'",
    curly_ldquote: '"',
    curly_rdquote: '"',
    em_dash: "-",
    double_plus: "++",
    ellipsis: "...",
    en_dash: "-",
    uml_I: "I",
    delta: "delta",
    AA: "AA",
    ue: "ue",
    degree: "degrees",
    delta2: "delta",
    one: "1",
    nine: "9",
    heart: "heart",
    fill_heart: "heart",
}


# Blacklisted unicode characters for each model type
blacklists = {
    "distilbert_base_multi_cased": [curly_lsquote, curly_rsquote, curly_ldquote, curly_rdquote, em_dash, double_plus, ellipsis, en_dash, uml_I, delta, AA, degree, ue],
    "minilm_l6": [delta2, degree, one, double_plus, one, nine, heart, fill_heart]
}


def detect_wrong_type(examples: dict, cols: list[str], dtype: Type) -> list[bool]:
    is_dtype = []
    # All columns should be of the same length
    for i in range(len(examples[cols[0]])):
        # Make sure entry is of the specified dtype for each column of interest
        is_dtype_ = all(isinstance(examples[col][i], dtype) for col in cols)
        is_dtype.append(is_dtype_)
    return is_dtype


def get_n_tokens_batched(examples: dict, text_col: list[str], tokenizer: PreTrainedTokenizerBase) -> dict:
    inputs = tokenizer(examples[text_col], truncation=False)
    n_tokens = [len(inp_ids) for inp_ids in inputs["input_ids"]]
    return {f"{text_col}_n_tokens": n_tokens}


def detect_unk_batched(examples: dict, cols: list[str], tokenizer: PreTrainedTokenizerBase) -> list[bool]:
    batch_ids_dict = {col: tokenizer(examples[col]).input_ids for col in cols}
    unk_markers = []
    # All columns should be of the same length
    for i in range(len(examples[cols[0]])):
        has_unk = any(tokenizer.unk_token_id in batch_ids_dict[col][i] for col in cols)
        unk_markers.append(has_unk)
        
    return unk_markers


def detect_only_unk_batched(examples: dict, cols: list[str], tokenizer: PreTrainedTokenizerBase) -> list[bool]:
    batch_ids_dict = {col: tokenizer(examples[col]).input_ids for col in cols}
    only_unk_markers = []
    
    # All columns should be of the same length
    for i in range(len(examples[cols[0]])):
        # Does *any* column (attribute) contain *only* unknown tokens?
        only_unk = any(all(token_id == tokenizer.unk_token_id for token_id in batch_ids_dict[col][i]) for col in cols)
        only_unk_markers.append(only_unk)
        
    return only_unk_markers


def get_blacklist_pattern(blacklist_key: str) -> re.Pattern:
    blacklist_pattern = re.compile("|".join(blacklists[blacklist_key]))
    return blacklist_pattern


def token_replacer(match):
    return blacklist_replace_dict[match.group(0)]


def replace_known_unk_tokens_batched(examples, cols, blacklist_pattern: re.Pattern) -> dict:
    replaced_batch = {col: [] for col in cols}

    # All columns should be of the same length
    for i in range(len(examples[cols[0]])):
        for col in cols:
            text = examples[col][i]
            replaced_text = blacklist_pattern.sub(token_replacer, html.unescape(text))
            replaced_batch[col].append(replaced_text)
            
    return replaced_batch


def replace_known_unk_tokens(examples, cols, blacklist_pattern: re.Pattern) -> dict:
    replaced = {}
    for col in cols:
        replaced[col] = blacklist_pattern.sub(token_replacer, html.unescape(examples[col]))
    return replaced
