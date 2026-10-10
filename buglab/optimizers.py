"""Search methods see only arrays and scalar fitness, not the field or beetle physics."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np

from .operator_story import GeneticOperation

class Optimizer(Protocol):
    def ask(self) -> np.ndarray: ...  # (population, genome_length)
    def tell(self, genomes: np.ndarray, scores: np.ndarray) -> None: ...

@dataclass
class SearchConfig:
    genome_length: int = 90
    population: int = 64
    max_turn: float = 0.42
    init_sigma: float = 0.11
    mutation_sigma: float = 0.17
    mutation_rate: float = 0.10
    tournament: int = 3
    elite: int = 2

    def __post_init__(self) -> None:
        if self.genome_length <= 0:
            raise ValueError("genome_length must be positive")
        if self.population <= 0:
            raise ValueError("population must be positive")
        if self.max_turn < 0 or self.init_sigma < 0 or self.mutation_sigma < 0:
            raise ValueError("turn and sigma values must be non-negative")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.tournament <= 0:
            raise ValueError("tournament must be positive")
        if not 0 <= self.elite <= self.population:
            raise ValueError("elite must be between zero and population")

class _Base:
    def __init__(self, cfg: SearchConfig, seed: int):
        self.cfg=cfg
        self.rng=np.random.default_rng(seed)
    def sample(self, n: int) -> np.ndarray:
        return np.clip(self.rng.normal(0,self.cfg.init_sigma,(n,self.cfg.genome_length)),
                       -self.cfg.max_turn,self.cfg.max_turn)

class RandomSearch(_Base):
    """Independent candidates; control for population size and evaluation budget."""
    def ask(self)->np.ndarray:
        return self.sample(self.cfg.population)
    def tell(self, genomes: np.ndarray, scores: np.ndarray)->None:
        pass

class GeneticAlgorithm(_Base):
    def __init__(self,cfg: SearchConfig,seed:int,*,trace_operations:bool=False,
                 trace_limit:int=256):
        super().__init__(cfg,seed)
        if trace_limit < 0:
            raise ValueError("trace_limit must be non-negative")
        self.seed=seed
        self.trace_operations=trace_operations
        self.trace_limit=trace_limit
        self.operation_log:list[GeneticOperation]=[]
        self._pending_operations:list[GeneticOperation]=[]
        self._generation=0
        self._next=None
        self._elite=None
        self._parents=None
        self._scores=None
        self.example_operation=None
        self.example_operation=None

    def _select_id(self)->int:
        ids=self.rng.integers(0,len(self._scores),size=self.cfg.tournament)
        return int(ids[np.argmax(self._scores[ids])])

    def _select(self)->np.ndarray:
        return self._parents[self._select_id()]

    def ask(self)->np.ndarray:
        if self._next is None: self._next=self.sample(self.cfg.population)
        return self._next.copy()

    def tell(self,genomes:np.ndarray,scores:np.ndarray)->None:
        scores=np.asarray(scores)
        if genomes.shape != (self.cfg.population,self.cfg.genome_length):
            raise ValueError("genomes do not match configured population and genome length")
        if scores.shape != (self.cfg.population,):
            raise ValueError("scores must contain one scalar per candidate")
        if self.trace_operations and self._pending_operations:
            for operation in self._pending_operations:
                if not np.array_equal(genomes[operation.candidate_index],operation.child):
                    raise ValueError("ask/tell candidate order changed during operation tracing")
                self.operation_log.append(
                    GeneticOperation(**{
                        **operation.__dict__,
                        "child_score":float(scores[operation.candidate_index]),
                    })
                )
            self._pending_operations=[]
        self._parents=genomes.copy(); self._scores=np.asarray(scores).copy()
        elite_ids=np.argsort(scores)[-self.cfg.elite:][::-1]
        children=[genomes[i].copy() for i in elite_ids]
        self.example_operation=None
        target_generation=self._generation+1
        pending=[]
        remaining=max(0,self.trace_limit-len(self.operation_log))
        if self.trace_operations:
            for candidate_index,parent_id in enumerate(elite_ids[:remaining]):
                parent=genomes[int(parent_id)].copy()
                pending.append(GeneticOperation(
                    self.seed,target_generation,target_generation*self.cfg.population+candidate_index,
                    candidate_index,"elite",int(parent_id),None,parent,None,
                    float(scores[parent_id]),None,None,parent.copy(),(),np.empty(0),parent.copy(),
                ))
        while len(children)<self.cfg.population:
            a_id=self._select_id(); b_id=self._select_id()
            a=self._parents[a_id]; b=self._parents[b_id]
            cut=int(self.rng.integers(1,self.cfg.genome_length))
            child=np.r_[a[:cut],b[cut:]].copy()
            # Mutate individual turning genes; also mutate short correlated segments
            # to allow smoothly changing a trajectory.
            before=child.copy()
            mask=self.rng.random(self.cfg.genome_length)<self.cfg.mutation_rate
            child[mask]+=self.rng.normal(0,self.cfg.mutation_sigma,mask.sum())
            if self.rng.random()<0.28:
                start=int(self.rng.integers(self.cfg.genome_length))
                width=int(self.rng.integers(2,9))
                child[start:min(start+width,self.cfg.genome_length)]+=self.rng.normal(0,0.095)
            child=np.clip(child,-self.cfg.max_turn,self.cfg.max_turn)
            candidate_index=len(children)
            if self.example_operation is None:
                self.example_operation={
                    'parent_a':a.copy(),'parent_b':b.copy(),'cut':cut,
                    'child':child.copy(),'before_mutation':before.copy(),
                    'mutated_indices':np.where(np.abs(child-before)>1e-12)[0].tolist(),
                }
            if self.trace_operations and len(pending)<remaining:
                mutated=np.where(np.abs(child-before)>1e-12)[0]
                pending.append(GeneticOperation(
                    self.seed,target_generation,target_generation*self.cfg.population+candidate_index,
                    candidate_index,"offspring",a_id,b_id,a.copy(),b.copy(),
                    float(scores[a_id]),float(scores[b_id]),cut,before.copy(),
                    tuple(int(index) for index in mutated),(child-before)[mutated].copy(),child.copy(),
                ))
            children.append(child)
        self._next=np.array(children)
        self._pending_operations=pending
        self._generation=target_generation

class LocalSearch(_Base):
    """Population-sized neighborhood around one incumbent; accepts only improvements."""
    def __init__(self,cfg:SearchConfig,seed:int):
        super().__init__(cfg,seed)
        self.incumbent=None; self.best_score=-np.inf
    def ask(self)->np.ndarray:
        if self.incumbent is None:
            return self.sample(self.cfg.population)
        out=np.tile(self.incumbent,(self.cfg.population,1))
        for row in out[1:]:
            mask=self.rng.random(self.cfg.genome_length)<self.cfg.mutation_rate
            row[mask]+=self.rng.normal(0,self.cfg.mutation_sigma,mask.sum())
            if self.rng.random()<0.4:
                start=int(self.rng.integers(self.cfg.genome_length))
                row[start:min(self.cfg.genome_length,start+int(self.rng.integers(2,9)))]+=self.rng.normal(0,0.11)
        return np.clip(out,-self.cfg.max_turn,self.cfg.max_turn)
    def tell(self,genomes:np.ndarray,scores:np.ndarray)->None:
        idx=int(np.argmax(scores))
        if scores[idx]>self.best_score:
            self.best_score=float(scores[idx]); self.incumbent=genomes[idx].copy()

METHODS={'ga':GeneticAlgorithm,'random':RandomSearch,'local':LocalSearch}
