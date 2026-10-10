"""Named map definitions, independent of physics and fitness policy."""

from dataclasses import dataclass

from .geometry import Rectangle


@dataclass(frozen=True)
class MapDefinition:
    """Static geometry needed to construct a :class:`buglab.environment.World`."""

    width: float
    height: float
    start: tuple[float, float]
    goal: tuple[float, float]
    obstacles: tuple[Rectangle, ...] = ()
    checkpoints: tuple[tuple[float, float], ...] = ()
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Map width and height must be positive")
        for label, point in (("start", self.start), ("goal", self.goal)):
            if len(point) != 2 or not (0 <= point[0] <= self.width and 0 <= point[1] <= self.height):
                raise ValueError(f"Map {label} must lie within its boundaries")
        for point in self.checkpoints:
            if len(point) != 2 or not (0 <= point[0] <= self.width and 0 <= point[1] <= self.height):
                raise ValueError("Map checkpoints must lie within its boundaries")

    def validate_clearance(self, radius: float) -> None:
        """Validate static point/obstacle clearance for a beetle radius."""
        if radius < 0:
            raise ValueError("Beetle radius cannot be negative")
        points = (("start", self.start), ("goal", self.goal), *(
            (f"checkpoint {index}", point) for index, point in enumerate(self.checkpoints)
        ))
        for label, (x, y) in points:
            if not (radius <= x <= self.width - radius and radius <= y <= self.height - radius):
                raise ValueError(f"Map {label} lacks boundary clearance for radius {radius}")
            if any(obstacle.contains(x, y, radius) for obstacle in self.obstacles):
                raise ValueError(f"Map {label} overlaps an obstacle for radius {radius}")
        for obstacle in self.obstacles:
            if not (0 <= obstacle.xmin < obstacle.xmax <= self.width and
                    0 <= obstacle.ymin < obstacle.ymax <= self.height):
                raise ValueError("Map obstacles must lie within map boundaries")


DEFAULT_MAP = MapDefinition(
    width=100.0,
    height=80.0,
    start=(8.0, 25.0),
    goal=(92.0, 60.0),
    obstacles=(Rectangle(35, 22, 46, 52), Rectangle(66, 43, 77, 72)),
    checkpoints=((30.0, 18.0), (50.0, 18.0), (62.0, 36.0), (81.0, 38.0)),
    name="default",
)

OPEN_FIELD = MapDefinition(
    width=100.0,
    height=80.0,
    start=(8.0, 40.0),
    goal=(92.0, 40.0),
    name="open_field",
)

TWO_BRIDGES = MapDefinition(
    width=100.0,
    height=80.0,
    start=(8.0, 40.0),
    goal=(92.0, 40.0),
    obstacles=(
        Rectangle(45.0, 0.0, 55.0, 18.0),
        Rectangle(45.0, 28.0, 55.0, 52.0),
        Rectangle(45.0, 62.0, 55.0, 80.0),
    ),
    checkpoints=((39.0, 23.0), (61.0, 23.0)),
    name="two_bridges",
)

NARROW_GATE = MapDefinition(
    width=100.0,
    height=80.0,
    start=(8.0, 18.0),
    goal=(92.0, 62.0),
    obstacles=(
        Rectangle(47.0, 0.0, 53.0, 37.0),
        Rectangle(47.0, 43.0, 53.0, 80.0),
    ),
    checkpoints=((42.0, 40.0), (58.0, 40.0)),
    name="narrow_gate",
)

DEAD_END = MapDefinition(
    width=100.0,
    height=80.0,
    start=(10.0, 40.0),
    goal=(90.0, 40.0),
    obstacles=(
        Rectangle(55.0, 20.0, 65.0, 60.0),
        Rectangle(30.0, 20.0, 55.0, 28.0),
        Rectangle(30.0, 52.0, 55.0, 60.0),
    ),
    checkpoints=((23.0, 15.0), (70.0, 15.0)),
    name="dead_end",
)

# One continuous rectangular wall curls inward. The short Euclidean direction
# crosses its lower/inner sections; the checkpoint sequence follows the free
# corridor around the curl.
SPIRAL = MapDefinition(
    width=72.0,
    height=60.0,
    start=(6.0, 5.0),
    goal=(40.0, 32.0),
    obstacles=(
        Rectangle(13.0, 9.0, 64.0, 14.0),
        Rectangle(59.0, 9.0, 64.0, 54.0),
        Rectangle(22.0, 49.0, 64.0, 54.0),
        Rectangle(22.0, 22.0, 27.0, 54.0),
        Rectangle(22.0, 22.0, 50.0, 27.0),
        Rectangle(45.0, 22.0, 50.0, 40.0),
        Rectangle(34.0, 35.0, 50.0, 40.0),
    ),
    checkpoints=((9.5, 18.0), (55.0, 18.0), (55.0, 45.0),
                 (30.0, 45.0), (30.0, 32.0)),
    name="spiral",
)


MAPS = {
    definition.name: definition
    for definition in (DEFAULT_MAP, OPEN_FIELD, TWO_BRIDGES, SPIRAL, NARROW_GATE, DEAD_END)
}


def get_map(name: str) -> MapDefinition:
    """Return a named map preset."""
    try:
        return MAPS[name]
    except KeyError as error:
        raise ValueError(f"Unknown map {name!r}; expected one of {tuple(MAPS)}") from error
