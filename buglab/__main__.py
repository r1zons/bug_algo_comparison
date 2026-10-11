"""Run: python -m buglab --generations 90 --population 64 --seeds 0 1 2"""
import argparse
from pathlib import Path
from .environment import World
from .fitness import FITNESS_NAMES,make_fitness
from .maps import MAPS,get_map
from .operator_story import (save_operation_log,select_operation_examples,
                             synthetic_operation)
from .optimizers import SearchConfig
from .experiment import run_method,save_history,save_outcome_comparison
from .visualization import (snapshots_figure,comparison_figure,save_evolution_gif,
                            moving_beetles_gif,crossover_figure,map_gallery,
                            fitness_outcomes_figure,synthetic_operation_figure,
                            real_operation_figure)

def main():
    p=argparse.ArgumentParser(description='Beetle evolution optimization laboratory')
    p.add_argument('--generations',type=int,default=90)
    p.add_argument('--population',type=int,default=64)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--methods',nargs='+',default=['ga','random','local'])
    p.add_argument('--map',choices=tuple(MAPS),default='default')
    p.add_argument('--fitness',choices=FITNESS_NAMES,default='geodesic')
    p.add_argument('--collision-penalty',type=float,default=0.5)
    p.add_argument('--smoothness-weight',type=float,default=1.0)
    p.add_argument('--gallery',action='store_true',help='render all named map presets')
    p.add_argument('--compare-fitness',nargs='+',choices=FITNESS_NAMES,
                   help='run the first method with each listed reward and export shared outcomes')
    p.add_argument('--operator-story',action='store_true',
                   help='record bounded GA operations and render teaching examples')
    p.add_argument('--operation-log-limit',type=int,default=256)
    p.add_argument('--output',default='outputs')
    p.add_argument('--gif',action='store_true')
    args=p.parse_args()
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    world=World.from_map(get_map(args.map))
    fitness=make_fitness(args.fitness,collision_penalty=args.collision_penalty,
                         smoothness_weight=args.smoothness_weight)
    cfg=SearchConfig(genome_length=world.steps,population=args.population,max_turn=world.max_turn)
    gsteps=tuple(range(0,args.generations,max(1,args.generations//14)))
    runs=[run_method(world,cfg,m,seed,args.generations,gsteps,fitness,
                     trace_operations=args.operator_story,
                     operation_trace_limit=args.operation_log_limit)
          for m in args.methods for seed in args.seeds]
    save_history(runs,folder)
    comparison_figure(runs,folder/'comparison.png')
    if args.gallery:
        map_gallery(folder/'map_gallery.png')
    ga=next((r for r in runs if r.method=='ga'),None)
    if ga:
        snapshots_figure(world,ga,folder/'ga_snapshots.png')
        if ga.example_operation:
            crossover_figure(world,ga.example_operation,folder/'crossover_example.png')
        if args.gif:
            save_evolution_gif(world,ga,folder/'ga_evolution.gif')
            moving_beetles_gif(world,ga,folder/'beetles_moving.gif')
    if args.operator_story:
        synthetic_operation_figure(synthetic_operation(),folder/'synthetic_crossover.png')
        if ga:
            save_operation_log(ga.operation_log,folder/'ga_operation_log.json')
            selected=select_operation_examples(ga.operation_log)
            for category,event in (
                ('improvement',selected.improvement),
                ('neutral',selected.neutral),
                ('degradation',selected.degradation),
            ):
                real_operation_figure(world,fitness,event,category,
                                      folder/f'real_operation_{category}.png')
                if event is None:
                    print(f'No {category} GA operation found in the bounded trace')
    for r in runs:
        print(f'{r.method:6} seed {r.seed:2} | map={r.map_name} fitness={r.fitness_name} '
              f'best={r.best_rollout.fitness:8.2f} '
              f'goal={r.best_rollout.reached} first_success_eval={r.first_success_evaluations} '
              f'first_success_batch_end={r.first_success_batch_end_evaluations}')
    if args.compare_fitness:
        comparison_runs=[
            run_method(world,cfg,args.methods[0],seed,args.generations,gsteps,
                       make_fitness(name,collision_penalty=args.collision_penalty,
                                    smoothness_weight=args.smoothness_weight))
            for name in args.compare_fitness for seed in args.seeds
        ]
        save_outcome_comparison(comparison_runs,folder/'fitness_comparison.csv')
        fitness_outcomes_figure(comparison_runs,folder/'fitness_comparison.png')
    print('Saved outputs:',folder.resolve())
if __name__=='__main__': main()
