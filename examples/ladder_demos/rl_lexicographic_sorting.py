import torch
from gfog.buffer import Buffer, Levels, Rung
from rich.console import Console

console = Console()


def main():
    # Hand-crafted samples (energy, distance) to showcase ladder behavior
    samples: list[tuple[float, float]] = [
        (28.0, 64.0),
        (18.0, 55.0),
        (4.0, 35.0),
        (4.0, 100.0),
    ]

    # Lexicographic levels: energy#1, distance#1, energy#2, distance#2, energy#3, distance#3
    # Clear semantics: minimize for upper bounds, maximize for lower bounds.
    energy_rung = Rung.minimize("energy", start=20.0, stop=5.0, num=3)
    # Demonstrate list-based thresholds for maximize
    distance_rung = Rung.maximize("distance", [30.0, 47.5])

    # Keep distance as the final open objective to continue optimizing it without a cap
    levels = Levels.ladder(
        [energy_rung, distance_rung], interleave=True, final_open="distance"
    )
    buffer = Buffer(buffer_size=4, value_levels=levels)

    # Insert samples into buffer; store raw values in the tensor for display.
    for idx, (energy, distance) in enumerate(samples):
        # Store raw for display; pass (energy, distance) directly — minimize/maximize handle semantics.
        state_tensor = torch.tensor([float(energy), float(distance), float(idx)])
        buffer.insert(state_tensor, [float(energy), float(distance)])

    buffer.print_values(slice(0, 4, 1))


if __name__ == "__main__":
    main()
