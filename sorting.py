import numpy as np
from typing import List
from pydantic import BaseModel, Field

# Example data following the EXACT same pattern as FAHP in app.py
sorting_example = {
    "tasks": [
        {"name": "Automated Robo-Advisory", "score": 1.94, "time": 6.0, "cost": 5.0},
        {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
        {"name": "Loyalty Level", "score": 4.02, "time": 6.0, "cost": 2.0},
        {"name": "Cross-Border Payments", "score": 1.16, "time": 6.0, "cost": 7.0},
        {"name": "Cryptocurrencies", "score": 1.11, "time": 6.0, "cost": 6.0},
        {"name": "Cross Sell", "score": 7.5, "time": 10.0, "cost": 6.0}
    ],
    "weights": {"w_time": 1.05, "w_cost": 1.05}
}

# Example data for Strategy 1 (Dataset 1)
strategy1_dataset1_example = {
    "tasks": [
        {"name": "Automated Robo-Advisory", "score": 1.94, "time": 6.0, "cost": 5.0},
        {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
        {"name": "Loyalty Level", "score": 4.02, "time": 6.0, "cost": 2.0},
        {"name": "Cross-Border Payments", "score": 1.16, "time": 6.0, "cost": 7.0},
        {"name": "Cryptocurrencies", "score": 1.11, "time": 6.0, "cost": 6.0},
        {"name": "Cross Sell", "score": 7.5, "time": 10.0, "cost": 6.0}
    ],
    "weights": {"w_time": 1.05, "w_cost": 1.05}
}

# Example data for Strategy 2 (Dataset 2)
strategy2_dataset2_example = {
    "tasks": [
        {"name": "Automated Robo-Advisory", "score": 1.94, "time": 4.0, "cost": 4.0},
        {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
        {"name": "Loyalty Level", "score": 4.02, "time": 6.0, "cost": 2.0},
        {"name": "Cross-Border Payments", "score": 1.16, "time": 4.0, "cost": 5.0},
        {"name": "Cryptocurrencies", "score": 1.11, "time": 3.0, "cost": 5.0},
        {"name": "Cross Sell", "score": 7.5, "time": 10.0, "cost": 6.0}
    ],
    "weights": {"w_time": 1.0, "w_cost": 1.0}
}

# Example for full comparison
comparison_example = {
    "strategy_1_dataset_1": strategy1_dataset1_example,
    "strategy_2_dataset_2": strategy2_dataset2_example
}


class SortingRequest(BaseModel):
    tasks: List[dict] = Field(
        default=sorting_example["tasks"],
        description="List of tasks with their strategic scores, time, and cost"
    )
    weights: dict = Field(
        default=sorting_example["weights"],
        description="Weights for time and cost factors"
    )


class StrategyComparisonRequest(BaseModel):
    strategy_1_dataset_1: dict = Field(
        default=comparison_example["strategy_1_dataset_1"],
        description="Strategy 1 with Dataset 1 (In-house approach)"
    )
    strategy_2_dataset_2: dict = Field(
        default=comparison_example["strategy_2_dataset_2"],
        description="Strategy 2 with Dataset 2 (Outsource approach)"
    )


class SortingResponse(BaseModel):
    pareto_optimal_tasks: List[dict]
    non_pareto_tasks: List[dict]
    min_possible_score: float
    max_possible_score: float
    average_time: float
    total_cost: float
    strategic_value_index: float


class StrategyComparisonResponse(BaseModel):
    strategy_1_results: dict
    strategy_2_results: dict
    comparison: List[dict]
    metrics: dict


def analyze_strategy(tasks, weights):
    """
    Analyze a strategy using Pareto optimality and multi-criteria decision analysis.
    Based on the correct algorithm from Sorting_Algo_phase3backup.py
    """
    task_names = [task["name"] for task in tasks]
    scores = np.array([task["score"] for task in tasks])
    times = np.array([task["time"] for task in tasks])
    costs = np.array([task["cost"] for task in tasks])
    w_time = weights["w_time"]
    w_cost = weights["w_cost"]

    # Function to check if task j dominates task i
    def dominates(i, j):
        return (w_time * times[j] <= w_time * times[i]) and (w_cost * costs[j] <= w_cost * costs[i]) and scores[j] >= scores[i]

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
    normalized_scores = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else np.zeros_like(scores)
    normalized_times = (times - times.min()) / (times.max() - times.min())
    normalized_costs = (costs - costs.min()) / (costs.max() - costs.min())

    # Assigning weights and combining scores (corrected formula from phase3backup)
    combined_scores_pareto = [(idx, -(normalized_scores[idx]) + w_time * (normalized_times[idx]) + w_cost * (normalized_costs[idx])) for idx in pareto_tasks]
    combined_scores_non_pareto = [(idx, -(normalized_scores[idx]) + w_time * (normalized_times[idx]) + w_cost * (normalized_costs[idx])) for idx in non_pareto_tasks]

    # Sorting the tasks
    sorted_pareto_tasks = sorted(combined_scores_pareto, key=lambda x: x[1], reverse=True)
    sorted_non_pareto_tasks = sorted(combined_scores_non_pareto, key=lambda x: x[1], reverse=True)

    # Calculate theoretical minimum and maximum scores (corrected from phase3backup)
    min_score = 0 + w_time * (1) + w_cost * (1)  # Theoretical minimum score
    max_score = -1 + w_time * (0) + w_cost * (0)  # Theoretical maximum score

    # Create Pareto results
    pareto_results = []
    for idx, score in sorted_pareto_tasks:
        pareto_results.append({
            "task_name": task_names[idx],
            "task_index": idx,
            "score": float(scores[idx]),
            "normalized_score": float(normalized_scores[idx]),
            "normalized_time": float(normalized_times[idx]),
            "normalized_cost": float(normalized_costs[idx]),
            "combined_score": float(score)
        })

    # Create non-Pareto results
    non_pareto_results = []
    for idx, score in sorted_non_pareto_tasks:
        non_pareto_results.append({
            "task_name": task_names[idx],
            "task_index": idx,
            "score": float(scores[idx]),
            "normalized_score": float(normalized_scores[idx]),
            "normalized_time": float(normalized_times[idx]),
            "normalized_cost": float(normalized_costs[idx]),
            "combined_score": float(score)
        })

    # Compute metrics
    average_time = float(np.mean(times))
    total_cost = float(np.sum(costs))
    all_scores = [score for _, score in combined_scores_pareto + combined_scores_non_pareto]
    strategic_value_index = float(np.sum(all_scores))

    return pareto_results, non_pareto_results, min_score, max_score, average_time, total_cost, strategic_value_index


def compare_strategies(strategy_1_data, strategy_2_data):
    """
    Compare two strategies and return detailed analysis.
    Based on the full workflow from Sorting_Algo_phase3backup.py
    """
    # Analyze both strategies
    strategy_1_results = analyze_strategy(strategy_1_data["tasks"], strategy_1_data["weights"])
    strategy_2_results = analyze_strategy(strategy_2_data["tasks"], strategy_2_data["weights"])

    # Create comparison results
    comparison = []
    task_names_1 = {task["task_name"]: task for task in strategy_1_results[0] + strategy_1_results[1]}
    task_names_2 = {task["task_name"]: task for task in strategy_2_results[0] + strategy_2_results[1]}

    all_task_names = set(task_names_1.keys()) | set(task_names_2.keys())

    for task_name in all_task_names:
        task_1 = task_names_1.get(task_name)
        task_2 = task_names_2.get(task_name)

        score_1 = task_1["combined_score"] if task_1 else None
        score_2 = task_2["combined_score"] if task_2 else None

        if score_1 is not None and score_2 is not None:
            comparison.append({
                "task_name": task_name,
                "strategy_1_score": score_1,
                "strategy_2_score": score_2,
                "score_difference": score_2 - score_1
            })
        elif score_1 is not None:
            comparison.append({
                "task_name": task_name,
                "strategy_1_score": score_1,
                "strategy_2_score": 0.0,
                "score_difference": -score_1
            })
        elif score_2 is not None:
            comparison.append({
                "task_name": task_name,
                "strategy_1_score": 0.0,
                "strategy_2_score": score_2,
                "score_difference": score_2
            })

    # Compute overall metrics
    metrics = {
        "strategy_1_avg_time": strategy_1_results[4],
        "strategy_1_total_cost": strategy_1_results[5],
        "strategy_1_strategic_index": strategy_1_results[6],
        "strategy_2_avg_time": strategy_2_results[4],
        "strategy_2_total_cost": strategy_2_results[5],
        "strategy_2_strategic_index": strategy_2_results[6]
    }

    return {
        "strategy_1_results": {
            "pareto_optimal_tasks": strategy_1_results[0],
            "non_pareto_tasks": strategy_1_results[1],
            "min_possible_score": strategy_1_results[2],
            "max_possible_score": strategy_1_results[3],
            "average_time": strategy_1_results[4],
            "total_cost": strategy_1_results[5],
            "strategic_value_index": strategy_1_results[6]
        },
        "strategy_2_results": {
            "pareto_optimal_tasks": strategy_2_results[0],
            "non_pareto_tasks": strategy_2_results[1],
            "min_possible_score": strategy_2_results[2],
            "max_possible_score": strategy_2_results[3],
            "average_time": strategy_2_results[4],
            "total_cost": strategy_2_results[5],
            "strategic_value_index": strategy_2_results[6]
        },
        "comparison": comparison,
        "metrics": metrics
    }
