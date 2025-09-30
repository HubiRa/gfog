from typing import Callable, Mapping, Sequence


class Rung:
    """Minimal objective spec using explicit rung intervals.

    - ranges: list of (lo, hi) with None meaning unbounded.
    - score: 0 if lo ≤ x ≤ hi, else distance to nearest bound.
    """

    def __init__(
        self, name: str, ranges: Sequence[tuple[float | None, float | None]]
    ) -> None:
        if not name:
            raise ValueError("Objective name must be non-empty")
        if not ranges:
            raise ValueError("Objective must define at least one rung range")
        _ranges: list[tuple[float | None, float | None]] = []
        for lo, hi in ranges:
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"Invalid range for {name}: low > high ({lo} > {hi})")
            _ranges.append((lo, hi))
        self.name = name
        self.ranges = _ranges
        # Optional direction marker for special rungs (e.g., open_last raw tie‑breaker)
        self._mode: str | None = None  # 'min' | 'max' | None

    @property
    def rung_count(self) -> int:
        return len(self.ranges)

    @staticmethod
    def _interval_score(x: float, lo: float | None, hi: float | None) -> float:
        if lo is None and hi is None:
            return 0.0
        if lo is not None and x < lo:
            return float(lo - x)
        if hi is not None and x > hi:
            return float(x - hi)
        return 0.0

    def score_for_rung(self, value: float, rung_index: int) -> float:
        lo, hi = self.ranges[rung_index]
        # Fully open rung: use raw value (or negated) as the score according to mode
        if lo is None and hi is None and self._mode in {"min", "max"}:
            return float(value) if self._mode == "min" else float(-value)
        return self._interval_score(value, lo, hi)

    # Convenience constructors using linspace-like generation
    @staticmethod
    def _linspace(start: float, stop: float, num: int) -> list[float]:
        if num < 1:
            raise ValueError("num must be >= 1")
        if num == 1:
            return [float(start)]
        step = (stop - start) / float(num - 1)
        return [float(start + i * step) for i in range(num)]

    @classmethod
    def linspace(
        cls,
        name: str,
        start: float,
        stop: float,
        num: int,
        *,
        open_last: bool = False,
    ) -> "Rung":
        """Generate upper-bound rungs (-inf, t_i] from thresholds t_i.

        Prefer using the semantic helpers `minimize`/`maximize` below.
        """
        thresholds = cls._linspace(start, stop, num)
        ranges = [(-float("inf"), t) for t in thresholds]
        if open_last and ranges:
            ranges.append((None, None))
        return cls(name, ranges)

    @classmethod
    def minimize(
        cls,
        name: str,
        start: float | Sequence[float],
        stop: float | None = None,
        num: int | None = None,
        *,
        open_last: bool = False,
    ) -> "Rung":
        """Minimize: generate upper-bound rungs (-inf, t_i].

        Usage:
        - Rung.minimize(name, start=20, stop=5, num=3, open_last=False)
        - Rung.minimize(name, [20, 15, 10, 5], open_last=False)

        If open_last=True, appends a final fully open rung that uses the raw value
        as a tie-breaker (smaller is better).
        """
        if isinstance(start, Sequence):
            if stop is not None or num is not None:
                raise ValueError(
                    "Provide either thresholds list or start/stop/num, not both"
                )
            thresholds = [float(t) for t in start]
        else:
            if stop is None or num is None:
                raise ValueError(
                    "When not providing a thresholds list, both stop and num are required"
                )
            thresholds = cls._linspace(float(start), float(stop), int(num))

        ranges = [(-float("inf"), t) for t in thresholds]
        if open_last:
            ranges.append((None, None))
        r = cls(name, ranges)
        r._mode = "min"
        return r

    @classmethod
    def maximize(
        cls,
        name: str,
        start: float | Sequence[float],
        stop: float | None = None,
        num: int | None = None,
        *,
        open_last: bool = False,
    ) -> "Rung":
        """Maximize: generate lower-bound rungs [t_i, +inf).

        Usage:
        - Rung.maximize(name, start=30, stop=65, num=3, open_last=False)
        - Rung.maximize(name, [30, 45, 65], open_last=False)

        If open_last=True, appends a final fully open rung that uses the negated value
        as a tie-breaker (larger original value is better).
        """
        if isinstance(start, Sequence):
            if stop is not None or num is not None:
                raise ValueError(
                    "Provide either thresholds list or start/stop/num, not both"
                )
            thresholds = [float(t) for t in start]
        else:
            if stop is None or num is None:
                raise ValueError(
                    "When not providing a thresholds list, both stop and num are required"
                )
            thresholds = cls._linspace(float(start), float(stop), int(num))

        ranges = [(t, float("inf")) for t in thresholds]
        if open_last:
            ranges.append((None, None))
        r = cls(name, ranges)
        r._mode = "max"
        return r


class Levels:
    def __init__(self, spec: int | Sequence[str]):
        if isinstance(spec, int):
            if spec < 1:
                raise ValueError("Number of levels must be >= 1")
            self._names = [f"L{i}" for i in range(spec)]
        elif isinstance(spec, Sequence) and all(isinstance(s, str) for s in spec):
            if len(set(spec)) != len(spec):
                raise ValueError("Level names must be unique")
            self._names = list(spec)
        else:
            raise TypeError(
                "Levels must be initialized with an int or a sequence of strings"
            )

        self._name_to_index = {name: idx for idx, name in enumerate(self._names)}
        # Optional expansion support (value ladder)
        self._transform: (
            Callable[[Sequence[float] | Mapping[str, float]], list[float]] | None
        ) = None
        self._objective_names: list[str] | None = None
        self._num_objectives: int | None = None

    def __getitem__(self, key: int | str) -> str | int:
        if isinstance(key, int):
            return self._names[key]
        elif isinstance(key, str):
            return self._name_to_index[key]
        raise TypeError("Key must be int or str")

    def num_levels(self) -> int:
        return len(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def index(self, name: str) -> int:
        return self._name_to_index[name]

    def names(self) -> list[str]:
        return self._names.copy()

    def __repr__(self) -> str:
        return f"Levels({self._names})"

    # Ladder support
    def has_transform(self) -> bool:
        return self._transform is not None

    @property
    def num_objectives(self) -> int | None:
        return self._num_objectives

    def transform(self, values: Sequence[float] | Mapping[str, float]) -> list[float]:
        if self._transform is None:
            raise RuntimeError("This Levels has no transform configured")
        return self._transform(values)

    def expand_input(self, value: float | Sequence[float]) -> list[float]:
        """Expand a scalar or short vector into the full ladder vector.

        Rules:
        - If value length already equals num_levels, pass through.
        - If a transform is configured and value length equals num_objectives, transform.
        - For scalar input, allow expansion only for single-objective ladders or single-level buffers.
        """
        n_levels = self.num_levels()

        # Scalar
        if isinstance(value, (int, float)):
            if self.has_transform():
                if (self.num_objectives or 0) == 1:
                    return self.transform([float(value)])
                raise ValueError(
                    "Scalar provided but ladder expects multiple objectives"
                )
            if n_levels != 1:
                raise ValueError(
                    f"Single value provided but buffer has {n_levels} levels"
                )
            return [float(value)]

        # Sequence
        seq = list(value)
        if len(seq) == n_levels:
            return [float(x) for x in seq]
        if self.has_transform():
            nobj = self.num_objectives or 0
            if len(seq) == nobj:
                return self.transform(seq)
            if len(seq) == 1 and nobj == 1:
                return self.transform(seq)

        raise ValueError(
            f"Value vector length {len(seq)} does not match levels ({n_levels}) and cannot be expanded via ladder"
        )

    @classmethod
    def ladder(
        cls,
        objectives: Sequence[Rung],
        interleave: bool = True,
        final_open: int | str | None = None,
    ) -> "Levels":
        if not objectives:
            raise ValueError("At least one objective required")
        rung_counts = {o.rung_count for o in objectives}
        # Allow uneven rung counts. We'll interleave up to max count and skip
        # objectives that don't have a rung at that index.
        n_rungs = max(rung_counts)

        # Build level names; label fully open final rung as "#open"
        def rung_name(obj: Rung, i: int) -> str:
            is_open_final = (
                i == obj.rung_count - 1
                and obj.ranges[i][0] is None
                and obj.ranges[i][1] is None
            )
            suffix = "open" if is_open_final else f"{i + 1}"
            return f"{obj.name}#{suffix}"

        if interleave:
            names: list[str] = []
            for i in range(n_rungs):
                for obj in objectives:
                    if i < obj.rung_count:
                        names.append(rung_name(obj, i))
        else:
            names = []
            for obj in objectives:
                for i in range(obj.rung_count):
                    names.append(rung_name(obj, i))

        # Optionally append a single final open rung for one chosen objective
        final_open_idx: int | None = None
        if final_open is not None:
            if isinstance(final_open, int):
                if not (0 <= final_open < len(objectives)):
                    raise IndexError(
                        f"final_open index {final_open} out of bounds for {len(objectives)} objectives"
                    )
                final_open_idx = final_open
            elif isinstance(final_open, str):
                try:
                    final_open_idx = [o.name for o in objectives].index(final_open)
                except ValueError:
                    raise ValueError(
                        f"final_open '{final_open}' not found among objective names {[o.name for o in objectives]}"
                    )
            else:
                raise TypeError("final_open must be int, str, or None")

            # Append the name for the final open rung
            names.append(f"{objectives[final_open_idx].name}#open")

        lvl = cls(names)
        lvl._objective_names = [o.name for o in objectives]
        lvl._num_objectives = len(objectives)

        def _tx(vals: Sequence[float] | Mapping[str, float]) -> list[float]:
            # Accept either positional (sequence) or mapping by objective name
            if isinstance(vals, Mapping):
                vec = [float(vals[name]) for name in lvl._objective_names or []]
            else:
                vec = list(map(float, vals))
                if len(vec) != (lvl._num_objectives or 0):
                    raise ValueError(
                        f"Expected {lvl._num_objectives} objective values, got {len(vec)}"
                    )

            scores: list[float] = []
            if interleave:
                # t1, u1, t2, u2, ... (skip if objective has fewer rungs)
                for i in range(n_rungs):
                    for j, obj in enumerate(objectives):
                        if i < obj.rung_count:
                            scores.append(obj.score_for_rung(vec[j], i))
            else:
                for j, obj in enumerate(objectives):
                    for i in range(obj.rung_count):
                        scores.append(obj.score_for_rung(vec[j], i))
            # Append final open rung score if configured
            if final_open_idx is not None:
                obj = objectives[final_open_idx]
                v = vec[final_open_idx]
                # If objective mode is known, use raw (min) or negated (max) value as tie-breaker
                if getattr(obj, "_mode", None) == "min":
                    scores.append(float(v))
                elif getattr(obj, "_mode", None) == "max":
                    scores.append(float(-v))
                else:
                    # Default to raw value if mode unknown
                    scores.append(float(v))
            return scores

        lvl._transform = _tx
        return lvl
