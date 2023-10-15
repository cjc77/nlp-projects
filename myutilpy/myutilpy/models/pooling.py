import torch


def pool_cls(embedding, attention_mask):
    # Extract the [CLS] embedding
    return embedding[:, 0, :]


def pool_mean(embedding, attention_mask):
    # Mean pool across dimension 1
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embedding.size()).float()
    pooled = torch.sum(embedding * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return pooled


pooling_fns = {
    "mean": pool_mean,
    "cls": pool_cls
}
