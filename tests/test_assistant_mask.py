from __future__ import annotations

from arlab.train.assistant_mask import find_all, mark_assistant_spans

START = [9, 8, 7]
END = [6, 5]


def test_find_all_returns_every_position():
    assert find_all([1, 2, 1, 2, 1], [1, 2]) == [0, 2]
    assert find_all([1, 1, 1], [2]) == []
    assert find_all([1, 2], []) == []


def test_marks_only_the_assistant_turn():
    # user ... START assistant ... END user ... START assistant ... END
    ids = [0, 0] + START + [10, 11] + END + [0, 0] + START + [12] + END
    mask = mark_assistant_spans(ids, START, END)

    assert sum(mask) == 2 + 2 + 1 + 2
    assert mask[: len([0, 0]) + len(START)] == [0] * (2 + len(START))
    # the tool/user region between the two turns stays untrained
    second_turn_start = len([0, 0]) + len(START) + 2 + len(END) + len([0, 0])
    assert mask[second_turn_start - 1] == 0


def test_unterminated_turn_is_marked_to_the_end():
    """A cut-off episode still trains on what the model did produce."""

    ids = [0] + START + [10, 11, 12]
    mask = mark_assistant_spans(ids, START, END)

    assert mask[0] == 0
    assert mask[-3:] == [1, 1, 1]


def test_no_assistant_turn_means_no_training_tokens():
    ids = [1, 2, 3, 4]
    assert mark_assistant_spans(ids, START, END) == [0, 0, 0, 0]


def test_empty_input_is_safe():
    assert mark_assistant_spans([], START, END) == []
    assert mark_assistant_spans([1, 2], [], []) == [0, 0]


def test_prompt_tokens_are_never_marked():
    """The whole point: the system prompt and tool observations must stay at zero."""

    system_and_user = [100, 101, 102, 103]
    ids = system_and_user + START + [10] + END
    mask = mark_assistant_spans(ids, START, END)

    assert all(bit == 0 for bit in mask[: len(system_and_user)])
    assert sum(mask) == 1 + len(END)
