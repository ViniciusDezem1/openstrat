import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
1. Scores are useless without sorting: Most important to least important
2. We need to know difference in the score in both strategy

"""

# Function to analyze a strategy on a given dataset
def analyze_strategy(strategy_name, task_names, s, t, c, w_time, w_cost):
    # Function to check if task j dominates task i
    def dominates(i, j):
        return (w_time * t[j] <= w_time * t[i]) and (w_cost * c[j] <= w_cost * c[i]) and s[j] >= s[i]

    # Function to merge two lists of tasks, keeping only the non-dominated ones
    def merge(tasks_left, tasks_right):
        merged_tasks = list(tasks_left)
        for task in tasks_right:
            dominated = False
            for check_task in tasks_left:
                if dominates(task, check_task):
                    dominated = True
                    break
            if not dominated:
                merged_tasks.append(task)
        return merged_tasks

    # Recursive function for "Divide and Conquer" approach
    def divide_and_conquer(l, r):
        if l == r:
            return [l]
        mid = (l + r) // 2
        left_tasks = divide_and_conquer(l, mid)
        right_tasks = divide_and_conquer(mid + 1, r)
        return merge(left_tasks, right_tasks)

    # Find Pareto optimal tasks
    pareto_tasks = divide_and_conquer(0, len(task_names) - 1)
    non_pareto_tasks = [idx for idx in range(len(task_names)) if idx not in pareto_tasks]

    # Normalize the scores to bring them in the range [0,1]
    normalized_s = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.zeros_like(s)
    normalized_t = (t - t.min()) / (t.max() - t.min())
    normalized_c = (c - c.min()) / (c.max() - c.min())
       

   #Assigning weights and combine scores
    combined_scores_pareto = [(idx, -(normalized_s[idx]) + w_time * (normalized_t[idx]) + w_cost * (normalized_c[idx])) for idx in pareto_tasks]
    combined_scores_non_pareto = [(idx, -(normalized_s[idx]) + w_time * (normalized_t[idx]) + w_cost * (normalized_c[idx])) for idx in non_pareto_tasks]

    #Sort tasks
    sorted_pareto_tasks = sorted(combined_scores_pareto, key=lambda x: x[1], reverse=True)
    sorted_non_pareto_tasks = sorted(combined_scores_non_pareto, key=lambda x: x[1], reverse=True)
    
    #Actual minimum and maximum scores
    all_scores = [score for _, score in combined_scores_pareto + combined_scores_non_pareto]
    min_score = min(all_scores)
    max_score = max(all_scores)    
    

    min_score = 0 + w_time * (1) + w_cost * (1) #Theoretical minimum score (worst preference score, highest time and cost)
    max_score = -1 + w_time * (0) + w_cost * (0)  #Theoretical maximum score (best preference score, lowest time and cost)  
    print("Minimum possible score:", min_score)
    print("Maximum possible score:", max_score)    


    # Print sorted tasks for the strategy
    print(f"{strategy_name} - Pareto Optimal Tasks from most to least optimal:")
    for idx, score in sorted_pareto_tasks:
        print(f"Task: {task_names[idx]}, Score: {score:.2f}")
    
    print("\nNon-Pareto Optimal Tasks from least to more optimal (but not Pareto optimal):")
    for idx, score in sorted_non_pareto_tasks:
        print(f"Task: {task_names[idx]}, Score: {score:.2f}")
    print("\n")
 
    
    return sorted_pareto_tasks, sorted_non_pareto_tasks, min_score, max_score


# Sample Data: Strategy 1
task_names = np.array([
    'Automated Robo-Advisory','Product Visibility','Loyalty Level','Cross-Border Payments',
    'Cryptocurrencies', 'Cross Sell'
])

s1 = np.array([1.94, 2.3, 4.02, 1.16, 1.11, 7.5])
t1 = np.array([6, 3, 6, 6, 6, 10])
c1 = np.array([5, 2, 2, 7, 6, 6]) 

# Sample Data: Strategy 2
s2 = np.array([1.94, 2.3, 4.02, 1.16, 1.11, 7.5])
t2 = np.array([4, 3, 6, 4, 3, 10])
c2 = np.array([4, 2, 2, 5, 5, 6]) 

#different weights for time and cost
strategies = {
    'Strategy 1': {'w_time': 1.05, 'w_cost': 1.05},
    'Strategy 2': {'w_time': 1, 'w_cost': 1} #Vinicius use 1 and 1 in real data scenario 
}

# Analyze each strategy for both datasets
for name, weights in strategies.items():
    print(f"{name} results for the first dataset:")
    analyze_strategy(name, task_names, s1, t1, c1, weights['w_time'], weights['w_cost'])
    print(f"{name} results for the second dataset:")
    analyze_strategy(name, task_names, s2, t2, c2, weights['w_time'], weights['w_cost'])


strategy_outcomes = {}
for name, weights in strategies.items():
    print(f"{name} results for the first dataset:")
    strategy_outcomes[name, 'Dataset 1'] = analyze_strategy(name, task_names, s1, t1, c1, weights['w_time'], weights['w_cost'])
    
    print(f"{name} results for the second dataset:")
    strategy_outcomes[name, 'Dataset 2'] = analyze_strategy(name, task_names, s2, t2, c2, weights['w_time'], weights['w_cost'])

# Compare the results of the two strategies
for task in task_names:
    idx = np.where(task_names == task)[0][0]
    
    score1_pareto = next((score for i, score in strategy_outcomes['Strategy 1', 'Dataset 1'][0] if i == idx), None)
    score2_pareto = next((score for i, score in strategy_outcomes['Strategy 2', 'Dataset 2'][0] if i == idx), None)
    
    # If a task is not in the Pareto optimal set, get the score from the non-Pareto set
    score1_non_pareto = next((score for i, score in strategy_outcomes['Strategy 1', 'Dataset 1'][1] if i == idx), None)
    score2_non_pareto = next((score for i, score in strategy_outcomes['Strategy 2', 'Dataset 2'][1] if i == idx), None)
    
    score1 = score1_pareto if score1_pareto is not None else score1_non_pareto
    score2 = score2_pareto if score2_pareto is not None else score2_non_pareto
    
    if score1 is not None and score2 is not None:
        print(f"Task: {task}, Strategy 1 Score: {score1:.2f}, Strategy 2 Score: {score2:.2f}, Difference: {score2 - score1:.2f}")
    elif score1 is not None:
        print(f"Task: {task}, Only Strategy 1 has a score: {score1:.2f}")
    elif score2 is not None:
        print(f"Task: {task}, Only Strategy 2 has a score: {score2:.2f}")
    else:
        print(f"Task: {task}, No score available in both strategies.")
        


# Function to compute average time and total cost for a strategy
def compute_metrics(t, c):
    average_time = np.mean(t)
    total_cost = np.sum(c)
    return average_time, total_cost



#########PLOT: SCORE SCORES in Each strategy
strategy_1_combined_scores = strategy_outcomes['Strategy 1', 'Dataset 1'][0] + strategy_outcomes['Strategy 1', 'Dataset 1'][1]
strategy_2_combined_scores = strategy_outcomes['Strategy 2', 'Dataset 2'][0] + strategy_outcomes['Strategy 2', 'Dataset 2'][1]

strategy_1_scores = [score for task, score in strategy_1_combined_scores]
strategy_2_scores = [score for task, score in strategy_2_combined_scores]
task_names_1 = [task_names[task] for task, _ in strategy_1_combined_scores]
task_names_2 = [task_names[task] for task, _ in strategy_2_combined_scores]

# Sorting the combined scores and task names
sorted_scores_1, sorted_task_names_1 = zip(*sorted(zip(strategy_1_scores, task_names_1), key=lambda x: x[0], reverse=True))
sorted_scores_2, sorted_task_names_2 = zip(*sorted(zip(strategy_2_scores, task_names_2), key=lambda x: x[0], reverse=True))

# Compute metrics
avg_time_1, total_cost_1 = compute_metrics(t1, c1)
avg_time_2, total_cost_2 = compute_metrics(t2, c2)
total_score_1 = np.sum(strategy_1_scores)
total_score_2 = np.sum(strategy_2_scores)

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
fig.suptitle('Phase 3', fontsize=13, fontweight='bold')

# Strategy 1
axes[0].barh(sorted_task_names_1, sorted_scores_1, color='#aec7e8')
axes[0].set_title(f'Strategy 1: In-house\nStrategic Value Index: {total_score_1:.2f}\nAvg Time: {avg_time_1:.2f}, Total Cost Scale: {total_cost_1:.2f}',fontsize=9.5)
axes[0].set_ylabel('Ranked Operational Actions')
axes[0].invert_yaxis()  # Highest scores at the top
axes[0].set_xlabel('Prioritization Value $(PV_i)$')
axes[0].set_xlim(-1, 2)

for i, v in enumerate(sorted_scores_1):
    sign = '+' if v > 0 else ''
    axes[0].text(v, i, f"{sign}{v:.2f}", color='black', va='center', fontweight='bold')

# Strategy 2
axes[1].barh(sorted_task_names_2, sorted_scores_2, color='#ff9896')
axes[1].set_title(f'Strategy 2: Outsource\nStrategic Value Index: {total_score_2:.2f}\nAvg Time: {avg_time_2:.2f}, Total Cost Scale: {total_cost_2:.2f}',fontsize=9.5)
axes[1].set_ylabel('Ranked Operational Actions')
axes[1].invert_yaxis()
axes[1].set_xlabel('Prioritization Value $(PV_i)$')
axes[1].set_xlim(-1, 2)

for i, v in enumerate(sorted_scores_2):
    sign = '+' if v > 0 else ''
    axes[1].text(v, i, f"{sign}{v:.2f}", color='black', va='center', fontweight='bold')



plt.savefig('output.png', dpi=1000)
axes[0].xaxis.grid(True, linestyle='-', linewidth=0.5)
axes[0].yaxis.grid(True, linestyle='-', linewidth=0.5)

axes[1].xaxis.grid(True, linestyle='-', linewidth=0.5)
axes[1].yaxis.grid(True, linestyle='-', linewidth=0.5)

# Adicionando um espaçamento entre os gráficos
plt.subplots_adjust(wspace=1.5)

# Salvar a figura com alta resolução na mesma pasta
output_path = 'output.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"Figura salva em: {output_path}")

plt.show()

