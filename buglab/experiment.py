"""Controlled benchmark engine with reward-independent outcome logging."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import numpy as np
from .environment import World,evaluate,Rollout
from .fitness import FitnessPolicy
from .operator_story import GeneticOperation,with_outcome
from .optimizers import METHODS,SearchConfig,GeneticAlgorithm

@dataclass
class Snapshot:
    generation:int
    evaluations:int
    best_fitness:float
    best_genome:np.ndarray
    best_rollout:Rollout
    top_paths:list[np.ndarray]
    candidate_genomes:np.ndarray | None = None
    candidate_scores:np.ndarray | None = None

@dataclass
class Run:
    method:str
    seed:int
    history:list[dict]
    snapshots:list[Snapshot]
    best_genome:np.ndarray
    best_rollout:Rollout
    first_success_evaluations:int | None
    example_operation:dict | None
    map_name:str = "default"
    fitness_name:str = "geodesic"
    search_config:dict = field(default_factory=dict)
    successful_evaluations:int = 0
    total_evaluations:int = 0
    environment_steps:int = 0
    best_success_rollout:Rollout | None = None
    operation_log:list[GeneticOperation] = field(default_factory=list)


def run_method(world:World, cfg:SearchConfig, method:str, seed:int,
               generations:int=90, snapshot_generations:tuple[int,...]=(0,5,15,30,60,89),
               fitness:FitnessPolicy | None=None,*,trace_operations:bool=False,
               operation_trace_limit:int=256)->Run:
    if method not in METHODS: raise ValueError(f'Unknown method {method}; expected {list(METHODS)}')
    if generations <= 0: raise ValueError('generations must be positive')
    if method == 'ga':
        optimizer=GeneticAlgorithm(cfg,seed,trace_operations=trace_operations,
                                   trace_limit=operation_trace_limit)
    else:
        optimizer=METHODS[method](cfg,seed)
    best=None; best_g=None; first_success=None; history=[]; snapshots=[]; example=None
    successful_evaluations=0; environment_steps=0; best_success=None
    operation_log=[]; trace_cursor=0
    fitness_name = getattr(fitness, "name", world.fitness_mode)
    for gen in range(generations):
        batch=optimizer.ask()
        results=[evaluate(world,genome,fitness) for genome in batch]
        scores=np.asarray([r.fitness for r in results])
        evals=(gen+1)*cfg.population
        for g,r in zip(batch,results):
            if best is None or r.fitness>best.fitness:
                best=r; best_g=g.copy()
            assert r.metrics is not None
            environment_steps += r.metrics.environment_steps
            if r.reached:
                successful_evaluations += 1
                if best_success is None or r.reached_at < best_success.reached_at:
                    best_success = r
        if first_success is None and any(r.reached for r in results):
            first_success=evals  # Batch-granular for all algorithms.
        outcome = best_success if best_success is not None else best
        assert outcome is not None and outcome.metrics is not None
        history.append({'method':method,'seed':seed,'generation':gen,
                        'map':world.map_name,'fitness':fitness_name,
                        'population':cfg.population,'genome_length':cfg.genome_length,
                        'evaluations':evals,'best_fitness':round(float(best.fitness),8),
                        'best_reached':int(best.reached),
                        'best_geodesic':round(float(best.min_geodesic),5),
                        'goal_reached':int(best_success is not None),
                        'steps_to_goal':outcome.metrics.steps_to_goal,
                        'collision_count':outcome.metrics.collision_count,
                        'actual_path_length':round(outcome.metrics.actual_path_length,8),
                        'final_distance':round(outcome.metrics.final_distance,8),
                        'success_rate':round(successful_evaluations/evals,8),
                        'environment_steps':environment_steps})
        if gen in snapshot_generations or gen==generations-1:
            ids=np.argsort(scores)[-10:]
            snapshots.append(Snapshot(gen,evals,best.fitness,best_g.copy(),best,
                                      [results[int(i)].positions.copy() for i in ids],
                                      batch.copy(), scores.copy()))
        optimizer.tell(batch,scores)
        if example is None and isinstance(optimizer,GeneticAlgorithm):
            example=optimizer.example_operation
        if isinstance(optimizer,GeneticAlgorithm) and trace_operations:
            completed=optimizer.operation_log[trace_cursor:]
            for operation in completed:
                metrics=results[operation.candidate_index].metrics
                assert metrics is not None
                operation_log.append(with_outcome(operation,metrics))
            trace_cursor=len(optimizer.operation_log)
    first_offspring=next((event for event in operation_log if event.kind=='offspring'),None)
    if example is None and first_offspring is not None:
        example={'parent_a':first_offspring.parent_a.copy(),
                 'parent_b':first_offspring.parent_b.copy(),
                 'cut':first_offspring.cut,'child':first_offspring.child.copy(),
                 'before_mutation':first_offspring.child_before_mutation.copy(),
                 'mutated_indices':list(first_offspring.mutated_indices)}
    return Run(method,seed,history,snapshots,best_g,best,first_success,example,
               world.map_name,fitness_name,asdict(cfg),successful_evaluations,
               generations*cfg.population,environment_steps,best_success,operation_log)


def save_history(runs:list[Run],folder:Path)->None:
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(runs[0].history[0]))
        writer.writeheader()
        for run in runs: writer.writerows(run.history)
    with (folder/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[
            'method','seed','map','fitness','population','genome_length','evaluations',
            'best_fitness','goal_reached','steps_to_goal','first_success_eval',
            'success_rate','collision_count','actual_path_length','final_distance',
            'environment_steps','best_geodesic',
        ])
        writer.writeheader()
        for r in runs:
            outcome = r.best_success_rollout or r.best_rollout
            assert outcome.metrics is not None
            writer.writerow({'method':r.method,'seed':r.seed,'map':r.map_name,
                             'fitness':r.fitness_name,
                             'population':r.search_config['population'],
                             'genome_length':r.search_config['genome_length'],
                             'evaluations':r.total_evaluations,
                             'best_fitness':r.best_rollout.fitness,
                             'goal_reached':int(r.successful_evaluations > 0),
                             'steps_to_goal':outcome.metrics.steps_to_goal,
                             'first_success_eval':r.first_success_evaluations,
                             'success_rate':r.successful_evaluations/r.total_evaluations,
                             'collision_count':outcome.metrics.collision_count,
                             'actual_path_length':outcome.metrics.actual_path_length,
                             'final_distance':outcome.metrics.final_distance,
                             'environment_steps':r.environment_steps,
                             'best_geodesic':r.best_rollout.min_geodesic,
                             })
            np.save(folder/f'best_genome_{r.method}_seed{r.seed}.npy',r.best_genome)


def save_outcome_comparison(runs:list[Run],path:Path)->None:
    """Export cross-fitness comparisons using shared outcomes only."""
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=['method','seed','map','fitness','evaluations','goal_reached','steps_to_goal',
            'success_rate','collision_count','actual_path_length','final_distance','environment_steps']
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for run in runs:
            outcome=run.best_success_rollout or run.best_rollout
            assert outcome.metrics is not None
            writer.writerow({
                'method':run.method,'seed':run.seed,'map':run.map_name,
                'fitness':run.fitness_name,'evaluations':run.total_evaluations,
                'goal_reached':int(run.successful_evaluations > 0),
                'steps_to_goal':outcome.metrics.steps_to_goal,
                'success_rate':run.successful_evaluations/run.total_evaluations,
                'collision_count':outcome.metrics.collision_count,
                'actual_path_length':outcome.metrics.actual_path_length,
                'final_distance':outcome.metrics.final_distance,
                'environment_steps':run.environment_steps,
            })
