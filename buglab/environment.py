"""Deterministic 2D beetle world and physics."""
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot, sin, cos, atan2, isfinite
import numpy as np

from .fitness import FitnessPolicy, LegacyFitness
from .geometry import Rectangle
from .maps import DEFAULT_MAP, MapDefinition
from .metrics import OutcomeMetrics, outcome_metrics

@dataclass
class World:
    width: float = DEFAULT_MAP.width
    height: float = DEFAULT_MAP.height
    start: tuple[float,float] = DEFAULT_MAP.start
    goal: tuple[float,float] = DEFAULT_MAP.goal
    obstacles: tuple[Rectangle,...] = DEFAULT_MAP.obstacles
    radius: float = 1.1
    goal_radius: float = 2.2
    step_length: float = 1.55
    steps: int = 90
    max_turn: float = 0.42  # radians per action
    grid_resolution: float = 1.0
    fitness_mode: str = "geodesic"  # also supports "euclidean" for reward-shaping experiments
    map_name: str | None = None
    checkpoints: tuple[tuple[float, float], ...] = ()
    _distances: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("World width and height must be positive")
        if self.radius < 0 or self.goal_radius < 0:
            raise ValueError("World radii must be non-negative")
        if self.step_length <= 0 or self.steps <= 0 or self.max_turn < 0 or self.grid_resolution <= 0:
            raise ValueError("Step length, steps and grid resolution must be positive; max_turn cannot be negative")
        if self.fitness_mode not in ("geodesic", "euclidean"):
            raise ValueError("fitness_mode must be geodesic or euclidean")
        self.obstacles = tuple(self.obstacles)
        self.checkpoints = tuple(self.checkpoints)
        if self.map_name is None:
            same_as_default = (
                self.width == DEFAULT_MAP.width and self.height == DEFAULT_MAP.height
                and self.start == DEFAULT_MAP.start and self.goal == DEFAULT_MAP.goal
                and self.obstacles == DEFAULT_MAP.obstacles
            )
            self.map_name = "default" if same_as_default else "custom"

    @classmethod
    def from_map(cls, definition: MapDefinition, **kwargs: object) -> "World":
        """Construct a world from pluggable geometry and optional physics settings."""
        world = cls(
            width=definition.width,
            height=definition.height,
            start=definition.start,
            goal=definition.goal,
            obstacles=definition.obstacles,
            checkpoints=definition.checkpoints,
            map_name=definition.name,
            **kwargs,
        )
        definition.validate_clearance(world.radius)
        return world

    def blocked_scalar(self,x:float,y:float,nx:float,ny:float)->bool:
        """Compatibility wrapper around the shared segment collision check."""
        return self.blocked(np.array((x, y)), np.array((nx, ny)))

    def blocked(self, p: np.ndarray, q: np.ndarray) -> bool:
        if not (self.radius <= q[0] <= self.width-self.radius and
                self.radius <= q[1] <= self.height-self.radius):
            return True
        return any(ob.hits_segment(p, q, self.radius) for ob in self.obstacles)

    def _make_distance_map(self) -> np.ndarray:
        """8-neighbour obstacle-aware goal distances for fitness shaping; not a path planner for beetles."""
        res = self.grid_resolution
        nx,ny = int(self.width/res)+1,int(self.height/res)+1
        blocked = np.zeros((ny,nx), dtype=bool)
        for j in range(ny):
            y=j*res
            for i in range(nx):
                x=i*res
                blocked[j,i]=(x<self.radius or x>self.width-self.radius or
                              y<self.radius or y>self.height-self.radius or
                              any(ob.contains(x,y,self.radius+0.3) for ob in self.obstacles))
        dist=np.full((ny,nx),np.inf)
        gx,gy=(int(round(self.goal[0]/res)),int(round(self.goal[1]/res)))
        if blocked[gy,gx]: raise ValueError('Goal is blocked')
        dist[gy,gx]=0.0
        heap=[(0.0, gy,gx)]
        neighbours=[(dj,di,res*hypot(di,dj)) for dj in (-1,0,1) for di in (-1,0,1) if di or dj]
        while heap:
            cur,j,i=heappop(heap)
            if cur>dist[j,i]+1e-12: continue
            for dj,di,w in neighbours:
                jj,ii=j+dj,i+di
                if 0<=jj<ny and 0<=ii<nx and not blocked[jj,ii]:
                    # Do not cut diagonally through corners.
                    if di and dj and (blocked[j,ii] or blocked[jj,i]): continue
                    nd=cur+w
                    if nd<dist[jj,ii]:
                        dist[jj,ii]=nd; heappush(heap,(nd,jj,ii))
        return dist

    def geodesic_xy(self,x:float,y:float)->float:
        if self._distances is None: self._distances=self._make_distance_map()
        res=self.grid_resolution
        i=int(round(x/res));j=int(round(y/res))
        if i<0:i=0
        elif i>=self._distances.shape[1]:i=self._distances.shape[1]-1
        if j<0:j=0
        elif j>=self._distances.shape[0]:j=self._distances.shape[0]-1
        v=self._distances[j,i]
        return float(v) if isfinite(v) else 500.0

    def geodesic(self, point: np.ndarray) -> float:
        if self._distances is None: self._distances=self._make_distance_map()
        i=int(np.clip(round(point[0]/self.grid_resolution),0,self._distances.shape[1]-1))
        j=int(np.clip(round(point[1]/self.grid_resolution),0,self._distances.shape[0]-1))
        value=float(self._distances[j,i])
        return value if np.isfinite(value) else 500.0

@dataclass
class Rollout:
    fitness: float
    positions: np.ndarray
    reached: bool
    reached_at: int | None
    collisions: int
    min_geodesic: float
    metrics: OutcomeMetrics | None = None


@dataclass
class EpisodeTrace:
    """Physics-only result, before a fitness policy is applied."""

    positions: np.ndarray
    reached: bool
    reached_at: int | None
    collisions: int
    turns: np.ndarray


def simulate(world: World, genome: np.ndarray) -> EpisodeTrace:
    """Run one deterministic episode without applying a training score."""
    if genome.shape != (world.steps,):
        raise ValueError(f'Expected genome of shape ({world.steps},), got {genome.shape}')
    if not np.all(np.isfinite(genome)):
        raise ValueError('Genome must be finite')
    x,y=world.start
    gx,gy=world.goal
    angle=atan2(gy-y,gx-x)
    trace=np.empty((world.steps+1,2), dtype=float)
    turns=np.empty(world.steps, dtype=float)
    trace[0]=(x,y)
    collisions=0; reached_at=None
    for t,turn in enumerate(genome):
        applied_turn=float(np.clip(turn,-world.max_turn,world.max_turn))
        turns[t]=applied_turn
        angle+=applied_turn
        nx=x+world.step_length*cos(angle)
        ny=y+world.step_length*sin(angle)
        if world.blocked_scalar(x,y,nx,ny):
            collisions+=1  # Turn in place; movement is blocked for this tick.
        else:
            x=nx;y=ny
        trace[t+1]=(x,y)
        if hypot(x-gx,y-gy)<=world.goal_radius:
            reached_at=t+1; break
    reached=reached_at is not None
    return EpisodeTrace(trace[:t+2], reached, reached_at, collisions, turns[:t+1])


def evaluate(world: World, genome: np.ndarray, fitness: FitnessPolicy | None = None) -> Rollout:
    """Simulate and score a genome with an optional pluggable fitness policy."""
    trace = simulate(world, genome)
    policy = fitness if fitness is not None else LegacyFitness(world.fitness_mode)
    scored = policy.score(world, trace)
    return Rollout(
        scored.fitness,
        trace.positions,
        trace.reached,
        trace.reached_at,
        trace.collisions,
        scored.min_distance,
        outcome_metrics(world, trace),
    )
