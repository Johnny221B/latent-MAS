import torch


def append_eos_token(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    eos_token_id: int,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """Append EOS immediately after the last valid token in each sequence."""
    if eos_token_id is None:
        raise ValueError("eos_token_id must be set to append EOS tokens")

    valid_lengths = attention_mask.sum(dim=1)
    output_len = int(valid_lengths.max().item()) + 1

    output_ids = torch.zeros(
        (input_ids.shape[0], output_len),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    output_mask = torch.zeros(
        (attention_mask.shape[0], output_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    for row_idx, valid_len in enumerate(valid_lengths.tolist()):
        valid_len = int(valid_len)
        valid_tokens = input_ids[row_idx][attention_mask[row_idx].bool()]
        if valid_len > 0:
            output_ids[row_idx, :valid_len] = valid_tokens
            output_mask[row_idx, :valid_len] = 1
        output_ids[row_idx, valid_len] = eos_token_id
        output_mask[row_idx, valid_len] = 1

    return output_ids, output_mask
