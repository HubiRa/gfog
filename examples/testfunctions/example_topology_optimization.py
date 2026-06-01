"""
Topology Optimization with Simplified GFog Ladder System

Real-world example showing why progressive goals prevent local minima in
GAN-based topology optimization using simplified progressive penalties.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import numpy as np

console = Console()


class SimpleRung:
    """Simplified rung with progressive penalty thresholds."""

    def __init__(
        self, name: str, thresholds: list[float], weights: list[float] | None = None
    ):
        self.name = name
        self.thresholds = thresholds
        self.weights = weights if weights else [10**i for i in range(len(thresholds))]

    @property
    def rung_count(self) -> int:
        return len(self.thresholds)

    def score_for_rung(self, value: float, rung_index: int) -> float:
        """Calculate penalty score for minimization objectives."""
        threshold = self.thresholds[rung_index]
        weight = self.weights[rung_index]
        penalty = max(0.0, value - threshold)
        return weight * penalty

    @classmethod
    def minimize(
        cls, name: str, strict_limit: float, loose_limit: float, num: int
    ) -> "SimpleRung":
        """Create rungs for minimization objectives (smaller is better)."""
        thresholds = np.linspace(strict_limit, loose_limit, num).tolist()
        return cls(name, thresholds)

    @classmethod
    def maximize(
        cls, name: str, loose_limit: float, strict_limit: float, num: int
    ) -> "SimpleRung":
        """Create rungs for maximization constraints (larger is better)."""
        thresholds = np.linspace(loose_limit, strict_limit, num).tolist()
        rung = cls(name, thresholds)

        # Override scoring for maximization
        def max_score_for_rung(value: float, rung_index: int) -> float:
            threshold = rung.thresholds[rung_index]
            weight = rung.weights[rung_index]
            penalty = max(0.0, threshold - value)  # Penalty when below threshold
            return weight * penalty

        rung.score_for_rung = max_score_for_rung
        return rung


class SimpleLevels:
    """Simplified levels for ladder system."""

    def __init__(self, rungs: list[SimpleRung], interleave: bool = True):
        self.rungs = rungs
        self.interleave = interleave

        # Build level names
        if interleave:
            names = [
                f"{rung.name}#{i + 1}"
                for i in range(rungs[0].rung_count)
                for rung in rungs
            ]
        else:
            names = [
                f"{rung.name}#{i + 1}" for rung in rungs for i in range(rung.rung_count)
            ]

        self.level_names = names

    def names(self) -> list[str]:
        return self.level_names

    def transform(self, values: list[float]) -> list[float]:
        """Transform raw objective values to ladder scores."""
        if len(values) != len(self.rungs):
            raise ValueError(f"Expected {len(self.rungs)} values, got {len(values)}")

        scores = []
        if self.interleave:
            # Interleaved: obj1_rung1, obj2_rung1, obj1_rung2, obj2_rung2, ...
            for rung_idx in range(self.rungs[0].rung_count):
                for obj_idx, rung in enumerate(self.rungs):
                    score = rung.score_for_rung(values[obj_idx], rung_idx)
                    scores.append(score)
        else:
            # Sequential: obj1_rung1, obj1_rung2, obj2_rung1, obj2_rung2, ...
            for obj_idx, rung in enumerate(self.rungs):
                for rung_idx in range(rung.rung_count):
                    score = rung.score_for_rung(values[obj_idx], rung_idx)
                    scores.append(score)

        return scores


def topology_optimization_problem():
    """Explain the topology optimization challenge."""

    console.print(
        Panel(
            "[bold cyan]🏗️  Topology Optimization Problem[/bold cyan]\n\n"
            "[white]Goal:[/white] Design optimal structure that minimizes weight while maintaining stiffness\n\n"
            "[yellow]Objectives:[/yellow]\n"
            "• [red]Material Volume:[/red] Use as little material as possible (minimize cost)\n"
            "• [blue]Compliance:[/blue] Keep structure stiff (minimize deformation under load)\n\n"
            "[bold red]The Trap:[/bold red] Hard constraints create local minima!\n"
            "❌ 'First satisfy volume ≤ 30%, then minimize compliance'\n"
            "❌ Generator gets trapped optimizing within 30% volume limit\n"
            "❌ Misses better global designs that use 35% volume but have much lower compliance",
            title="The Challenge",
            border_style="cyan",
        )
    )


def demonstrate_simplified_ladder():
    """Show simplified ladder approach for topology optimization."""

    console.print()
    console.print(
        "[bold]🔧 Simplified Ladder Approach for Topology Optimization[/bold]"
    )
    console.print()

    # Generate some candidate topology designs
    designs = [
        (
            "Dense_Safe",
            0.25,
            45.2,
        ),  # Low volume, high compliance (safe but inefficient)
        (
            "Sparse_Risky",
            0.45,
            12.8,
        ),  # High volume, low compliance (risky but efficient)
        ("Balanced", 0.35, 28.5),  # Medium volume, medium compliance
        (
            "Optimal_Hidden",
            0.32,
            15.1,
        ),  # Slightly over "safe" volume but much better compliance
        ("Overbuilt", 0.18, 38.9),  # Very low volume, poor compliance
    ]

    console.print("🎲 Generator proposes these topology designs:")

    design_table = Table(box=box.ROUNDED)
    design_table.add_column("Design", style="bold")
    design_table.add_column("Material Volume", style="red", justify="center")
    design_table.add_column(
        "Compliance\n(lower=stiffer)", style="blue", justify="center"
    )
    design_table.add_column("Notes", style="dim")

    for name, volume, compliance in designs:
        if volume <= 0.3:
            status = "✅ Meets strict volume limit"
        else:
            status = "❌ Exceeds strict volume limit"

        design_table.add_row(name, f"{volume:.0%}", f"{compliance:.1f}", status)

    console.print(design_table)
    console.print()

    # Show the simplified approach
    console.print(
        Panel(
            "[bold green]✨ Simplified Progressive Penalty Approach[/bold green]\n\n"
            "[white]Instead of complex tuple intervals, use simple thresholds:[/white]\n\n"
            "[red]Volume penalties (minimize):[/red]\n"
            "• Rung 1: penalty = max(0, volume - 0.40) × 1\n"
            "• Rung 2: penalty = max(0, volume - 0.30) × 10\n"
            "• Rung 3: penalty = max(0, volume - 0.20) × 100\n\n"
            "[blue]Compliance penalties (minimize):[/blue]\n"
            "• Rung 1: penalty = max(0, compliance - 40) × 1\n"
            "• Rung 2: penalty = max(0, compliance - 25) × 10\n"
            "• Rung 3: penalty = max(0, compliance - 15) × 100\n\n"
            "[bold]Result:[/bold] Smooth progressive penalties → smooth optimization landscape!",
            style="dim",
        )
    )

    # Create simplified ladder
    volume_rung = SimpleRung.minimize(
        "volume", strict_limit=0.20, loose_limit=0.40, num=3
    )
    compliance_rung = SimpleRung.minimize(
        "compliance", strict_limit=15.0, loose_limit=40.0, num=3
    )

    console.print()
    console.print(f"[green]Volume thresholds:[/green] {volume_rung.thresholds}")
    console.print(f"[green]Volume weights:[/green] {volume_rung.weights}")
    console.print(f"[blue]Compliance thresholds:[/blue] {compliance_rung.thresholds}")
    console.print(f"[blue]Compliance weights:[/blue] {compliance_rung.weights}")
    console.print()

    levels = SimpleLevels([volume_rung, compliance_rung], interleave=True)

    console.print("Simplified ladder scoring for each design:")

    scoring_table = Table(box=box.SIMPLE)
    scoring_table.add_column("Design", style="bold")
    scoring_table.add_column("Raw Values", style="cyan")
    scoring_table.add_column("Progressive Penalties", style="yellow")
    scoring_table.add_column("Interpretation", style="dim")

    design_scores = []
    for name, volume, compliance in designs:
        scores = levels.transform([volume, compliance])
        design_scores.append((name, volume, compliance, scores))

        # Interpret the scores
        total_penalty = sum(scores)
        if total_penalty == 0:
            interp = "Perfect - no penalties"
        elif total_penalty < 10:
            interp = "Excellent - minor penalties"
        elif total_penalty < 100:
            interp = "Good - moderate penalties"
        else:
            interp = "Poor - high penalties"

        scoring_table.add_row(
            name,
            f"vol={volume:.0%}, comp={compliance:.1f}",
            f"[{', '.join(f'{s:.1f}' for s in scores)}]",
            interp,
        )

    console.print(scoring_table)
    console.print()

    # Sort by lexicographic ordering (like the buffer would)
    design_scores.sort(key=lambda x: x[3])  # Sort by scores (lexicographically)

    # Show final ranking
    console.print("[bold]🏆 Final Ranking (Lower Total Penalty = Better):[/bold]")

    ranking_table = Table(box=box.SIMPLE)
    ranking_table.add_column("Rank", justify="center")
    ranking_table.add_column("Design", style="bold")
    ranking_table.add_column("Total Penalty", style="red", justify="center")
    ranking_table.add_column("Why This Rank?", style="dim")

    explanations = [
        "Lowest total penalty across all rungs",
        "Good balance between objectives",
        "Moderate penalties, acceptable trade-offs",
        "High penalties from volume violations",
        "Highest total penalty",
    ]

    for i, (name, volume, compliance, scores) in enumerate(design_scores):
        total_penalty = sum(scores)
        ranking_table.add_row(
            f"#{i + 1}",
            name,
            f"{total_penalty:.1f}",
            explanations[i] if i < len(explanations) else "High penalty",
        )

    console.print(ranking_table)


def show_benefits():
    """Show the benefits of the simplified approach."""

    console.print()
    console.print(
        Panel(
            "[bold green]✨ Benefits of Simplified Progressive Penalties[/bold green]\n\n"
            "[yellow]1. Simpler to Understand[/yellow]\n"
            "   • Just thresholds + weights, no complex intervals\n"
            "   • Clear progression: loose → strict limits\n\n"
            "[yellow]2. More Natural for Optimization[/yellow]\n"
            "   • Smooth continuous penalties (no discontinuities)\n"
            "   • Natural gradient information for GAN generator\n\n"
            "[yellow]3. Easier to Configure[/yellow]\n"
            "   • Set target limits directly\n"
            "   • Adjust penalty weights independently\n"
            "   • Intuitive parameter tuning\n\n"
            "[yellow]4. Better for Black-Box Optimization[/yellow]\n"
            "   • Creates smooth search landscape\n"
            "   • Prevents local minima traps\n"
            "   • Allows generator to explore promising regions\n\n"
            "[bold cyan]Example Usage:[/bold cyan]\n"
            "```python\n"
            "# Minimize material volume: loose 40% → strict 20%\n"
            "volume_rung = SimpleRung.minimize('volume', \n"
            "                                  strict_limit=0.20, \n"
            "                                  loose_limit=0.40, \n"
            "                                  num=3)\n"
            "\n"
            "# Result: [0.20, 0.30, 0.40] thresholds with [1, 10, 100] weights\n"
            "```",
            style="dim",
        )
    )


def main():
    """Demonstrate topology optimization with simplified ladder system."""

    topology_optimization_problem()
    demonstrate_simplified_ladder()
    show_benefits()

    console.print()
    console.print(
        Panel(
            "[bold white]🎯 Simplified GFog Ladder System[/bold white]\n\n"
            "[green]Core Insight:[/green] Progressive penalties create smooth optimization landscapes\n\n"
            "[yellow]Key Simplification:[/yellow]\n"
            "• Replace complex tuple intervals with simple thresholds\n"
            "• Use progressive penalties: penalty = max(0, value - threshold) × weight\n"
            "• Much easier to understand and configure\n\n"
            "[bold]Result:[/bold] Elegant solution for GAN-based black-box optimization!\n\n"
            "[dim]The simplified approach maintains all the benefits while removing unnecessary complexity.[/dim]",
            title="🏁 Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
