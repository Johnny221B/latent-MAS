import torch

from src.utils.token_utils import append_eos_token


def test_append_eos_token_appends_after_last_valid_answer_token():
    answer_ids = torch.tensor(
        [
            [7, 8],
            [9, 0],
        ],
        dtype=torch.long,
    )
    answer_mask = torch.tensor(
        [
            [1, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )

    extended_ids, extended_mask = append_eos_token(
        input_ids=answer_ids,
        attention_mask=answer_mask,
        eos_token_id=2,
    )

    assert torch.equal(
        extended_ids,
        torch.tensor(
            [
                [7, 8, 2],
                [9, 2, 0],
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        extended_mask,
        torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )
