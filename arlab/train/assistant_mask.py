"""Mark exactly the tokens the model generated, without trusting the chat template.

Why this exists: Qwen3's template has no ``{% generation %}`` block, so
``apply_chat_template(return_assistant_tokens_mask=True)`` returns nothing and the
fallback marked every token trainable. Training on prompt and tool-observation
tokens pushed KL from 0.007 to 10 over 200 steps and made the policy worse.

An earlier attempt reconstructed the mask by rendering the prefix with
``add_generation_prompt=True`` and diffing lengths. That works on hand-built
conversations but returned an empty mask on real trajectories -- probably because
the tool-response turns are re-rendered differently when they are last. Rather than
reason about template internals, mark the spans directly: the assistant turns are
the regions between the assistant header and the end-of-turn marker, both of which
are plain token sequences that can be located by scanning ``input_ids``.

The scanning function is pure (ids in, mask out), so it is tested without a
tokenizer or a GPU.
"""

from __future__ import annotations

from typing import Sequence


def find_all(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    """Every start index where ``needle`` occurs in ``haystack``."""

    if not needle:
        return []
    hits: list[int] = []
    width = len(needle)
    for index in range(len(haystack) - width + 1):
        if list(haystack[index : index + width]) == list(needle):
            hits.append(index)
    return hits


def mark_assistant_spans(
    ids: Sequence[int],
    start_ids: Sequence[int],
    end_ids: Sequence[int],
) -> list[int]:
    """Mask of 0/1 over ``ids``: 1 inside every assistant turn, end marker included.

    An unterminated final turn (the model was cut off) is marked to the end, which
    is what actually happened -- hiding it would silently drop the truncated tail.
    """

    mask = [0] * len(ids)
    starts = find_all(ids, start_ids)
    if not starts:
        return mask

    width = len(start_ids)
    for start in starts:
        body = start + width
        ends = find_all(ids[body:], end_ids)
        stop = body + ends[0] + len(end_ids) if ends else len(ids)
        for index in range(body, min(stop, len(ids))):
            mask[index] = 1
    return mask


def assistant_token_ids(
    tokenizer,
    header: str = "<|im_start|>assistant\n",
    end: str = "<|im_end|>\n",
) -> tuple[list[int], list[int]]:
    """Tokenise the two markers once, so scanning is pure integer comparison."""

    return (
        list(tokenizer.encode(header, add_special_tokens=False)),
        list(tokenizer.encode(end, add_special_tokens=False)),
    )
