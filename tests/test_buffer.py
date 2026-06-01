import torch

from gfog.buffer import Buffer, Levels, Rung


def test_buffer_insert_sort_len_and_getitem() -> None:
    buf = Buffer(3)
    for tensor, value in [
        (torch.tensor([1.0]), 3.0),
        (torch.tensor([2.0]), 1.0),
        (torch.tensor([3.0]), 2.0),
        (torch.tensor([4.0]), 0.5),
    ]:
        buf.insert(tensor, value)

    assert len(buf) == 3
    assert torch.equal(buf[0], torch.tensor([4.0]))
    assert torch.equal(buf[1], torch.tensor([2.0]))
    assert buf.get_value(0) == 0.5
    assert buf.get_sorted_values() == [[0.5], [1.0], [2.0]]


def test_buffer_insert_many_accepts_row_and_column_major() -> None:
    levels = Levels(["a", "b"])

    row_tensors = [torch.tensor([1.0]), torch.tensor([2.0])]
    row_major = Buffer(4, value_levels=levels)
    row_major.insert_many(row_tensors, [[1.0, 10.0], [0.5, 20.0]])
    assert row_major.get_sorted_values() == [[0.5, 20.0], [1.0, 10.0]]

    column_tensors = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    column_major = Buffer(4, value_levels=levels)
    column_major.insert_many(
        column_tensors,
        [[1.0, 0.5, 2.0], [10.0, 20.0, 5.0]],
    )
    assert column_major.get_sorted_values() == [
        [0.5, 20.0],
        [1.0, 10.0],
        [2.0, 5.0],
    ]


def test_buffer_levels_ladder_transform() -> None:
    levels = Levels.ladder(
        [
            Rung.minimize("constraint", [5.0, 2.0]),
            Rung.minimize("fx", [10.0, 1.0], open_last=True),
        ]
    )
    buf = Buffer(4, value_levels=levels)
    buf.insert(torch.tensor([1.0, 2.0]), [1.5, 3.0])

    assert buf.get_sorted_values() == [[0.0, 0.0, 0.0, 2.0, 3.0]]


def test_buffer_clear_removes_contents_without_dead_slot_attribute() -> None:
    buf = Buffer(2)
    buf.insert(torch.tensor([1.0]), 1.0)
    buf.clear()

    assert len(buf) == 0
    assert not hasattr(buf, "next_free_slot")


def test_buffer_get_bottom_k_zero_returns_empty() -> None:
    buf = Buffer(3)
    buf.insert(torch.tensor([1.0]), 1.0)
    buf.insert(torch.tensor([2.0]), 2.0)

    out_k = buf.get_bottom_k(0)
    out_p = buf.get_bottom_p(0.0)

    assert out_k.shape == (0, 1)
    assert out_p.shape == (0, 1)


def test_buffer_clear_resets_tensor_shape_metadata() -> None:
    buf = Buffer(2)
    buf.insert(torch.tensor([1.0]), 1.0)
    buf.clear()
    buf.insert(torch.tensor([1.0, 2.0]), 1.0)

    assert len(buf) == 1
    assert torch.equal(buf[0], torch.tensor([1.0, 2.0]))
