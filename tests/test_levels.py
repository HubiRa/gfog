from gfog.buffer import Levels, Rung


def test_ladder_interleave_and_open_last() -> None:
    levels = Levels.ladder(
        [
            Rung.minimize("loss", [5.0, 1.0]),
            Rung.maximize("score", [2.0, 4.0], open_last=True),
        ],
        interleave=True,
    )

    assert levels.names() == [
        "loss#1",
        "score#1",
        "loss#2",
        "score#2",
        "score#open",
    ]
    assert levels.expand_input([0.5, 5.0]) == [0.0, 0.0, 0.0, 0.0, -5.0]


def test_ladder_accepts_mapping_input() -> None:
    levels = Levels.ladder(
        [Rung.minimize("fx", [10.0, 1.0]), Rung.maximize("acc", [0.8, 0.9])]
    )

    assert levels.transform({"fx": 0.5, "acc": 0.95}) == [0.0, 0.0, 0.0, 0.0]
