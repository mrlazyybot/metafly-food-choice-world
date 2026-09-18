from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'src'))
from metafly.core import experiment, FOODS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

out=Path(__file__).parent/'results'; summary,runs=experiment(out)
foods=sorted(summary,key=lambda x:summary[x]['mean_value'],reverse=True)
fig=plt.figure(figsize=(12,7),facecolor='#f7f3e8')
gs=fig.add_gridspec(2,2,width_ratios=[1.25,1])
ax=fig.add_subplot(gs[:,0]); ax.set_facecolor('#fffaf0'); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
ax.set_title('MetaFly food-choice world',fontsize=18,weight='bold')
colors={'banana':'#f5d742','apple':'#dc3d4b','orange':'#f39c34','grape':'#8457a6','tomato':'#d64949','carrot':'#e9872d','broccoli':'#4f914c','cucumber':'#69a96b'}
pos=[(2,8),(5,8),(8,8),(2,5),(8,5),(2,2),(5,2),(8,2)]
for food,(x,y) in zip(FOODS,pos):
 ax.add_patch(Circle((x,y),.6,color=colors[food],alpha=.9)); ax.text(x,y-.95,food,ha='center',fontsize=11)
ax.add_patch(Circle((5,5),.36,color='#efc43a',ec='black')); ax.plot([4.55,5.45],[5,5],color='black'); ax.text(5,4.25,'agent',ha='center',weight='bold')
ax.text(5,.55,'Illustrative world. Food rewards are synthetic hypotheses, not observed fly preferences.',ha='center',fontsize=9)
ax2=fig.add_subplot(gs[0,1]); vals=[summary[x]['mean_value'] for x in foods]; ax2.barh(foods[::-1],vals[::-1],color=[colors[x] for x in foods[::-1]]); ax2.set_xlim(0,1); ax2.set_title('Learned value after 240 trials\n(mean across 20 seeded runs)'); ax2.set_xlabel('value')
ax3=fig.add_subplot(gs[1,1]); rates=[summary[x]['top_rank_rate'] for x in foods]; ax3.barh(foods[::-1],rates[::-1],color=[colors[x] for x in foods[::-1]]); ax3.set_xlim(0,1); ax3.set_title('Fraction of runs ranked first'); ax3.set_xlabel('rate')
fig.tight_layout(); fig.savefig(out/'metafly_results.png',dpi=180)
print(out/'metafly_results.png')
