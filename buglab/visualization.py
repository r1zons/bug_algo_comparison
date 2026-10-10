"""Only graphics. Consumes saved result objects, never influences search."""
from pathlib import Path
import textwrap
from matplotlib import pyplot as plt
from matplotlib import patches, animation
import numpy as np
from .environment import World
from .experiment import Run
from .fitness import FitnessPolicy
from .maps import MAPS
from .operator_story import GeneticOperation,SyntheticOperation


def field(ax,world:World,show_checkpoints:bool=False)->None:
    ax.set_xlim(0,world.width);ax.set_ylim(0,world.height)
    ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y')
    for ob in world.obstacles:
        ax.add_patch(patches.Rectangle((ob.xmin,ob.ymin),ob.xmax-ob.xmin,ob.ymax-ob.ymin,
                                       fc='#9daab4',ec='#697988',alpha=.72))
    ax.scatter(*world.start,s=85,marker='o',color='#3d6ca3',zorder=8)
    ax.scatter(*world.goal,s=160,marker='*',color='#da923c',zorder=8)
    if show_checkpoints and world.checkpoints:
        checkpoints=np.asarray(world.checkpoints)
        ax.scatter(checkpoints[:,0],checkpoints[:,1],s=34,marker='D',
                   color='#7e57a3',zorder=7,label='Ordered checkpoints')
    ax.annotate('Start',world.start,xytext=(4,7),textcoords='offset points',fontsize=9)
    ax.annotate('Goal',world.goal,xytext=(7,7),textcoords='offset points',fontsize=9)
    ax.grid(alpha=.12)


def map_gallery(path:Path)->None:
    """Render every named preset with consistent styling."""
    columns=3
    rows=(len(MAPS)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(15,8.5),constrained_layout=True)
    for ax,(name,definition) in zip(axes.flat,MAPS.items()):
        world=World.from_map(definition)
        field(ax,world,show_checkpoints=True)
        ax.set_title(name.replace('_',' ').title())
    for ax in axes.flat[len(MAPS):]:
        ax.set_visible(False)
    fig.suptitle('Bug Evolution Lab map presets',fontsize=17)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def fitness_outcomes_figure(runs:list[Run],path:Path)->None:
    """Compare reward strategies using outcomes, never cross-reward fitness."""
    if not runs:
        raise ValueError('At least one run is required')
    columns=['Fitness','Seed','Success rate','Steps to goal','Collisions',
             'Path length','Final distance','Evaluations']
    cells=[]
    for run in runs:
        outcome=run.best_success_rollout or run.best_rollout
        assert outcome.metrics is not None
        cells.append([
            run.fitness_name,str(run.seed),f'{run.successful_evaluations/run.total_evaluations:.3f}',
            '—' if outcome.metrics.steps_to_goal is None else str(outcome.metrics.steps_to_goal),
            str(outcome.metrics.collision_count),f'{outcome.metrics.actual_path_length:.1f}',
            f'{outcome.metrics.final_distance:.1f}',str(run.total_evaluations),
        ])
    fig,ax=plt.subplots(figsize=(12,max(2.6,0.5*len(cells)+1.5)))
    ax.axis('off')
    table=ax.table(cellText=cells,colLabels=columns,cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.4)
    ax.set_title(f'Fitness guidance on {runs[0].map_name}: shared physical outcomes only',pad=16)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def snapshots_figure(world:World,run:Run,path:Path)->None:
    selected=[run.snapshots[0],run.snapshots[len(run.snapshots)//3],
              run.snapshots[2*len(run.snapshots)//3],run.snapshots[-1]]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,s in zip(axes.flat,selected):
        field(ax,world)
        for p in s.top_paths: ax.plot(p[:,0],p[:,1],color='#9a9cb4',alpha=.37,lw=.95)
        p=s.best_rollout.positions
        ax.plot(p[:,0],p[:,1],color='#10877b',lw=2.7)
        ax.set_title(f'Поколение {s.generation} · best={s.best_fitness:.1f} · goal={"да" if s.best_rollout.reached else "нет"}',fontsize=11)
    fig.suptitle('Эволюция траекторий жуков — GA',fontsize=17)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def comparison_figure(runs:list[Run],path:Path)->None:
    fig,ax=plt.subplots(figsize=(9,5))
    labels={'ga':'Генетический алгоритм','random':'Случайный поиск','local':'Локальный поиск'}
    colors={'ga':'#148b7e','random':'#bc863a','local':'#5c77ad'}
    for m in dict.fromkeys(r.method for r in runs):
        relevant=[r for r in runs if r.method==m]
        y=np.asarray([[v['best_fitness'] for v in r.history] for r in relevant])
        x=[v['evaluations'] for v in relevant[0].history]
        ax.plot(x,np.median(y,axis=0),label=labels[m],color=colors[m],lw=2)
        if len(relevant)>1:
            ax.fill_between(x,np.percentile(y,25,axis=0),np.percentile(y,75,axis=0),
                            color=colors[m],alpha=.12)
    ax.set_xlabel('Оценки fitness (одинаковый бюджет)')
    ax.set_ylabel('Лучший найденный fitness')
    ax.set_title('GA и базовые алгоритмы: результат от бюджета')
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def save_evolution_gif(world:World,run:Run,path:Path,frames:int=55)->None:
    """Real stored best trajectories by generation; not simulated illustration."""
    from PIL import Image
    fig,ax=plt.subplots(figsize=(8,5.8));field(ax,world)
    line,=ax.plot([],[],lw=2.8,color='#10877b')
    title=ax.set_title('')
    paths=[(s.generation,s.best_rollout.positions,s.best_rollout.reached) for s in run.snapshots]
    # Snapshot-to-snapshot playback; increasing number of snapshots makes GIF smoother.
    def frame(k):
        gen,p,ok=paths[min(int(k*len(paths)/frames),len(paths)-1)]
        line.set_data(p[:,0],p[:,1])
        title.set_text(f'GA: поколение {gen} · цель {"достигнута" if ok else "не достигнута"}')
        return line,title
    ani=animation.FuncAnimation(fig,frame,frames=frames,interval=100,blit=False)
    ani.save(path,writer=animation.PillowWriter(fps=8))
    plt.close(fig)


def moving_beetles_gif(world:World, run:Run, path:Path, frames_each:int=16)->None:
    """Time-lapse: ten real candidates actually walking their stored paths, across generations."""
    from matplotlib import animation
    chosen=[run.snapshots[i] for i in [0, min(2,len(run.snapshots)-1),
                                         len(run.snapshots)//2, len(run.snapshots)-1]]
    fig,ax=plt.subplots(figsize=(8,5.9))
    field(ax,world)
    tracks=[ax.plot([],[],lw=1.4,alpha=.45,color='#8694b0')[0] for _ in range(10)]
    dots=[ax.plot([],[],marker='o',ms=5,color='#5a78a1')[0] for _ in range(10)]
    title=ax.set_title('')
    total=len(chosen)*frames_each
    def draw(k):
        which=min(k//frames_each,len(chosen)-1)
        step_frac=(k%frames_each+1)/frames_each
        ss=chosen[which]
        for i,(line,dot) in enumerate(zip(tracks,dots)):
            if i<len(ss.top_paths):
                traj=ss.top_paths[i]
                n=max(1,int(step_frac*(len(traj)-1)))
                line.set_data(traj[:n+1,0],traj[:n+1,1])
                dot.set_data([traj[n,0]],[traj[n,1]])
            else:
                line.set_data([],[]);dot.set_data([],[])
        title.set_text(f'Поколение {ss.generation} · шаг {int(step_frac*world.steps)} / {world.steps}')
        return (*tracks,*dots,title)
    ani=animation.FuncAnimation(fig,draw,frames=total,interval=95,blit=False)
    ani.save(path,writer=animation.PillowWriter(fps=10))
    plt.close(fig)


def crossover_figure(world:World, operation:dict, path:Path)->None:
    """A real selected parent pair / one-point crossover / pointwise mutation."""
    a=operation['parent_a'];b=operation['parent_b'];child=operation['child']
    cut=operation['cut']
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),constrained_layout=True)
    for ax,g,title,color in zip(axes,(a,b,child),('Родитель A','Родитель B','Потомок'),
                                ('#5b7dad','#bd915b','#158f7e')):
        field(ax,world)
        # Import at call-site to keep renderer distinct from simulator semantics.
        from .environment import evaluate
        path_points=evaluate(world,g).positions
        ax.plot(path_points[:,0],path_points[:,1],color=color,lw=2.2)
        ax.set_title(title)
    fig.suptitle(f'Реальное скрещивание: точка разреза {cut}/{len(a)}, затем {len(operation["mutated_indices"])} изменённых углов')
    fig.savefig(path,dpi=160)
    plt.close(fig)


PARENT_A_COLOR='#3973ac'
PARENT_B_COLOR='#dd8b2c'
MUTATION_COLOR='#c23b75'
CHILD_COLOR='#168579'


def synthetic_operation_figure(example:SyntheticOperation,path:Path)->None:
    """Render a compact gene-level explanation with angles displayed in degrees."""
    genomes=(example.parent_a,example.parent_b,example.child_before_mutation,example.child)
    labels=('Parent A','Parent B','Child after crossover','Final child')
    rows=(3,2,1,0)
    fig,ax=plt.subplots(figsize=(12,4.8))
    mutated=set(example.mutated_indices)
    for genome,label,row in zip(genomes,labels,rows):
        for index,value in enumerate(np.rad2deg(genome)):
            if label=='Parent A': color=PARENT_A_COLOR
            elif label=='Parent B': color=PARENT_B_COLOR
            else: color=PARENT_A_COLOR if index<example.cut else PARENT_B_COLOR
            if label=='Final child' and index in mutated: color=MUTATION_COLOR
            ax.add_patch(patches.Rectangle((index-.45,row-.31),.9,.62,facecolor=color,
                                            edgecolor='white',linewidth=1.5))
            ax.text(index,row,f'{value:+.0f}°',ha='center',va='center',color='white',
                    fontsize=10,fontweight='bold')
        ax.text(-.65,row,label,ha='right',va='center',fontsize=11,fontweight='bold')
    ax.axvline(example.cut-.5,color='#30343b',lw=2,ls='--')
    ax.text(example.cut-.5,3.55,f'one-point cut: {example.cut}',ha='center',fontsize=10)
    mutation_text=', '.join(
        f'gene {index}: {np.rad2deg(example.child_before_mutation[index]):+.0f}° '
        f'→ {np.rad2deg(example.child[index]):+.0f}°'
        for index in example.mutated_indices
    )
    ax.text((len(example.child)-1)/2,-.65,f'Mutations: {mutation_text}',ha='center',
            color=MUTATION_COLOR,fontsize=10,fontweight='bold')
    ax.text((len(example.child)-1)/2,-1.02,
            'Crossover combines control instructions—not geometrically connected trajectory segments.',
            ha='center',fontsize=10)
    ax.set_xticks(range(len(example.child)),[str(index) for index in range(len(example.child))])
    ax.set_xlabel('Gene index (relative turn angles; displayed in degrees, stored in radians)')
    ax.set_xlim(-2.2,len(example.child)-.3);ax.set_ylim(-1.25,3.85)
    ax.set_yticks([]);ax.set_frame_on(False)
    ax.set_title('Deterministic crossover and mutation example',fontsize=16,pad=16)
    fig.tight_layout();fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)


def real_operation_figure(world:World,fitness:FitnessPolicy,event:GeneticOperation | None,
                          category:str,path:Path)->None:
    """Replay and render one real logged operation, or state that none was observed."""
    if event is None:
        fig,ax=plt.subplots(figsize=(11,4.8));ax.axis('off')
        ax.text(.5,.58,f'No {category} event recorded',ha='center',va='center',
                fontsize=20,fontweight='bold',transform=ax.transAxes)
        ax.text(.5,.42,'The bounded log is reported honestly; no example is synthesized.',
                ha='center',va='center',fontsize=12,transform=ax.transAxes)
        fig.savefig(path,dpi=180,facecolor='white',bbox_inches='tight');plt.close(fig)
        return
    if event.kind!='offspring' or event.parent_b is None or event.parent_b_score is None:
        raise ValueError('Real operation figures require a crossover offspring event')
    from .environment import evaluate
    parent_a=evaluate(world,event.parent_a,fitness)
    parent_b=evaluate(world,event.parent_b,fitness)
    child=evaluate(world,event.child,fitness)
    expected=(event.parent_a_score,event.parent_b_score,event.child_score)
    replayed=(parent_a.fitness,parent_b.fitness,child.fitness)
    if event.child_score is None or not np.allclose(replayed,expected,rtol=0,atol=1e-10):
        raise ValueError('Operation replay does not match its recorded scores')

    fig=plt.figure(figsize=(14,8.2),facecolor='white',constrained_layout=True)
    grid=fig.add_gridspec(2,3,height_ratios=(2.2,1.2))
    for column,(rollout,title,color,score) in enumerate((
        (parent_a,f'Parent A · candidate {event.parent_a_id}',PARENT_A_COLOR,event.parent_a_score),
        (parent_b,f'Parent B · candidate {event.parent_b_id}',PARENT_B_COLOR,event.parent_b_score),
        (child,f'Child · candidate {event.candidate_index}',CHILD_COLOR,event.child_score),
    )):
        ax=fig.add_subplot(grid[0,column]);field(ax,world)
        ax.plot(rollout.positions[:,0],rollout.positions[:,1],color=color,lw=2.6)
        ax.set_title(f'{title}\nfitness {score:.3f}',fontsize=11)
    genes=fig.add_subplot(grid[1,:])
    indices=np.arange(len(event.child))
    before_degrees=np.rad2deg(event.child_before_mutation)
    child_degrees=np.rad2deg(event.child)
    genes.scatter(indices[:event.cut],before_degrees[:event.cut],s=24,color=PARENT_A_COLOR,
                  label='Inherited from A',zorder=3)
    genes.scatter(indices[event.cut:],before_degrees[event.cut:],s=24,color=PARENT_B_COLOR,
                  label='Inherited from B',zorder=3)
    if event.mutated_indices:
        changed=np.asarray(event.mutated_indices)
        genes.vlines(changed,before_degrees[changed],child_degrees[changed],
                     color=MUTATION_COLOR,lw=1.8,alpha=.75)
        genes.scatter(changed,child_degrees[changed],marker='*',s=90,color=MUTATION_COLOR,
                      label='Final mutated value',zorder=4)
    genes.axvline(event.cut-.5,color='#30343b',ls='--',lw=1.5,
                  label=f'Crossover cut {event.cut}')
    genes.set_xlabel('Gene index');genes.set_ylabel('Relative turn (degrees)')
    genes.grid(alpha=.18);genes.legend(ncol=4,loc='upper right',fontsize=9)
    delta=event.child_score-event.better_parent_score
    mutations=', '.join(
        f'{index}: Δ{np.rad2deg(change):+.1f}°'
        for index,change in zip(event.mutated_indices,event.mutation_deltas)
    ) or 'none'
    genes.set_title(textwrap.fill(f'Applied mutation changes: {mutations}',width=150),
                    fontsize=9,loc='left',pad=8)
    metrics=event.child_metrics
    outcome='' if metrics is None else (
        f' · goal={metrics.goal_reached}, collisions={metrics.collision_count}, '
        f'path={metrics.actual_path_length:.1f}'
    )
    fig.suptitle(
        f'{category.title()} · seed {event.seed}, generation {event.generation}, '
        f'operation {event.operation_id}\nchild − better parent = {delta:+.3f}{outcome}\n'
        f'cut={event.cut} · {len(event.mutated_indices)} mutated genes',fontsize=14,
    )
    fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)
