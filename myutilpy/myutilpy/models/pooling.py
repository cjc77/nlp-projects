import torch


def pool_cls(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Performs CLS pooling (summarization via CLS token) for a given batch of embeddings.

    Args:
        embeddings (torch.Tensor): A batch of token embeddings.
        attention_mask (torch.Tensor): The embeddings' corresponding attention mask.

    Returns:
        torch.Tensor: A batch of pooled embeddings.
    """
    # Extract the [CLS] embedding
    return embeddings[:, 0, :]


def pool_mean(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Performs mean pooling (summarization via token-wise mean) for a given embedding.

    Args:
        embeddings (torch.Tensor): A batch of token embeddings.
        attention_mask (torch.Tensor): The embeddings' corresponding attention mask.

    Returns:
        torch.Tensor: A batch of pooled embeddings.
    """
    # Mean pool across dimension 1
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    pooled = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return pooled


# Key-indexed pooling function
pooling_fns = {
    "mean": pool_mean,
    "cls": pool_cls
}
