from typing import List

import numpy as np
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

# Example data for multiple scenarios as in Sorting_Algo_phase3backup.py
scenarios_example = {
    "scenarios": [
        {
            "name": "Strategy 1 - Dataset 1",
            "description": "In-house approach with original data",
            "tasks": [
                {"name": "Automated Robo-Advisory", "score": 1.94, "time": 6.0, "cost": 5.0},
                {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
                {"name": "Loyalty Level", "score": 4.02, "time": 6.0, "cost": 2.0},
                {"name": "Cross-Border Payments", "score": 1.16, "time": 6.0, "cost": 7.0},
                {"name": "Cryptocurrencies", "score": 1.11, "time": 6.0, "cost": 6.0},
                {"name": "Cross Sell", "score": 7.5, "time": 10.0, "cost": 6.0}
            ],
            "weights": {"w_time": 1.05, "w_cost": 1.05}
        },
        {
            "name": "Strategy 1 - Dataset 2",
            "description": "In-house approach with modified data",
            "tasks": [
                {"name": "Automated Robo-Advisory", "score": 1.94, "time": 6.0, "cost": 5.0},
                {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
                {"name": "Loyalty Level", "score": 4.02, "time": 6.0, "cost": 2.0},
                {"name": "Cross-Border Payments", "score": 1.16, "time": 6.0, "cost": 7.0},
                {"name": "Cryptocurrencies", "score": 1.11, "time": 6.0, "cost": 6.0},
                {"name": "Cross Sell", "score": 7.5, "time": 10.0, "cost": 6.0}
            ],
            "weights": {"w_time": 1.05, "w_cost": 1.05}
        },
        {
            "name": "Strategy 2 - Dataset 1",
            "description": "Outsource approach with original data",
            "tasks": [
                {"name": "Automated Robo-Advisory", "score": 1.94, "time": 4.0, "cost": 4.0},
                {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
                {"name": "Loyalty Level", "score": 4.02, "time": 6.0, "cost": 2.0},
                {"name": "Cross-Border Payments", "score": 1.16, "time": 4.0, "cost": 5.0},
                {"name": "Cryptocurrencies", "score": 1.11, "time": 3.0, "cost": 5.0},
                {"name": "Cross Sell", "score": 7.5, "time": 10.0, "cost": 6.0}
            ],
            "weights": {"w_time": 1.0, "w_cost": 1.0}
        },
        {
            "name": "Strategy 2 - Dataset 2",
            "description": "Outsource approach with modified data",
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
    ]
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


class ScenariosRequest(BaseModel):
    scenarios: List[dict] = Field(
        default=scenarios_example["scenarios"],
        description="List of scenarios, each containing name, description, tasks, and weights"
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


class ScenarioResult(BaseModel):
    scenario_name: str
    scenario_description: str
    pareto_optimal_tasks: List[dict]
    non_pareto_tasks: List[dict]
    min_possible_score: float
    max_possible_score: float
    average_time: float
    total_cost: float
    strategic_value_index: float


class ScenariosResponse(BaseModel):
    scenarios: List[dict]
    comparison: List[dict]
    overall_metrics: dict


class StrategyComparisonResponse(BaseModel):
    strategy_1_results: dict
    strategy_2_results: dict
    comparison: List[dict]
    metrics: dict


def generate_single_strategy_summary(pareto_results, non_pareto_results, min_score, max_score, average_time, total_cost, strategic_value_index):
    """
    Generate a summary for a single strategy analysis.
    """
    total_tasks = len(pareto_results) + len(non_pareto_results)
    pareto_percentage = round((len(pareto_results) / total_tasks) * 100, 1) if total_tasks > 0 else 0

    # Find best and worst tasks
    all_tasks = pareto_results + non_pareto_results
    best_task = max(all_tasks, key=lambda x: x["svi"])
    worst_task = min(all_tasks, key=lambda x: x["svi"])

    # Generate narrative summary
    narrative = f"""
    Single Strategy Analysis Summary:

    This analysis evaluated {total_tasks} tasks using Pareto optimality and multi-criteria decision analysis.
    The strategy achieved an overall Strategic Value Index (SVI) of {strategic_value_index:.2f}.

    Key Performance Metrics:
    • Strategic Value Index: {strategic_value_index:.2f}
    • Average Time Requirement: {average_time:.2f} units
    • Total Cost: {total_cost:.2f} units
    • Pareto Optimal Tasks: {len(pareto_results)} out of {total_tasks} ({pareto_percentage}%)

    Task Performance Highlights:
    • Best Performing Task: {best_task['task_name']} (SVI: {best_task['svi']:.2f})
    • Worst Performing Task: {worst_task['task_name']} (SVI: {worst_task['svi']:.2f})
    • Performance Range: {best_task['svi']:.2f} to {worst_task['svi']:.2f} SVI points

    The analysis identified {len(pareto_results)} Pareto optimal tasks that represent the most efficient
    combinations of strategic value, time, and cost considerations.
    """

    summary = {
        "analysis_overview": {
            "total_tasks_analyzed": total_tasks,
            "analysis_method": "Pareto Optimality with Multi-Criteria Decision Analysis",
            "scoring_formula": "SVI = normalized_score + w_time * (-normalized_time) + w_cost * (-normalized_cost)"
        },
        "narrative_summary": narrative.strip(),
        "strategy_performance": {
            "strategic_value_index": strategic_value_index,
            "average_time": average_time,
            "total_cost": total_cost,
            "pareto_optimal_tasks": len(pareto_results),
            "non_pareto_tasks": len(non_pareto_results),
            "pareto_percentage": pareto_percentage
        },
        "key_findings": [
            {
                "finding": "Best Performing Task",
                "task": best_task["task_name"],
                "svi": best_task["svi"],
                "description": f"{best_task['task_name']} achieved the highest SVI of {best_task['svi']:.2f}"
            },
            {
                "finding": "Worst Performing Task",
                "task": worst_task["task_name"],
                "svi": worst_task["svi"],
                "description": f"{worst_task['task_name']} achieved the lowest SVI of {worst_task['svi']:.2f}"
            },
            {
                "finding": "Pareto Optimality",
                "value": f"{len(pareto_results)}/{total_tasks} tasks ({pareto_percentage}%)",
                "description": f"{len(pareto_results)} out of {total_tasks} tasks are Pareto optimal"
            }
        ],
        "performance_summary": f"Strategy achieved SVI={strategic_value_index:.2f}, Avg Time={average_time:.2f}, Total Cost={total_cost:.2f}, Pareto Tasks={len(pareto_results)}/{total_tasks}"
    }

    return summary


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

    # Assigning weights and combining scores (corrected formula from phase_3.py)
    # Score should be positive, time and cost should be negative deflators
    combined_scores_pareto = [(idx, (normalized_scores[idx]) + w_time * (-normalized_times[idx]) + w_cost * (-normalized_costs[idx])) for idx in pareto_tasks]
    combined_scores_non_pareto = [(idx, (normalized_scores[idx]) + w_time * (-normalized_times[idx]) + w_cost * (-normalized_costs[idx])) for idx in non_pareto_tasks]

    # Sorting the tasks
    sorted_pareto_tasks = sorted(combined_scores_pareto, key=lambda x: x[1], reverse=True)
    sorted_non_pareto_tasks = sorted(combined_scores_non_pareto, key=lambda x: x[1], reverse=True)

    # Calculate theoretical minimum and maximum scores (corrected from phase_3.py)
    min_score = 0 + w_time * (-1) + w_cost * (-1)  # Theoretical minimum score (worst preference score, highest time and cost)
    max_score = 1 + w_time * (0) + w_cost * (0)  # Theoretical maximum score (best preference score, lowest time and cost)

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
            "svi": float(score)
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
            "svi": float(score)
        })

    # Compute metrics
    average_time = float(np.mean(times))
    total_cost = float(np.sum(costs))
    all_scores = [score for _, score in combined_scores_pareto + combined_scores_non_pareto]
    strategic_value_index = float(np.sum(all_scores))

    return pareto_results, non_pareto_results, min_score, max_score, average_time, total_cost, strategic_value_index


def generate_summary(scenario_results, comparison):
    """
    Generate a summary of the analysis results.
    Based on the summary format from Sorting_Algo_phase3backup.py
    """
    summary = {
        "analysis_overview": {
            "total_scenarios_analyzed": len(scenario_results),
            "total_tasks_analyzed": len(comparison),
            "analysis_method": "Pareto Optimality with Multi-Criteria Decision Analysis",
            "scoring_formula": "SVI = normalized_score + w_time * (-normalized_time) + w_cost * (-normalized_cost)"
        },
        "narrative_summary": "",
        "key_findings": [],
        "scenario_summaries": []
    }

    # Generate key findings
    best_scenario = max(scenario_results, key=lambda x: x["strategic_value_index"])
    worst_scenario = min(scenario_results, key=lambda x: x["strategic_value_index"])

    summary["key_findings"].append({
        "finding": "Best Performing Scenario",
        "scenario": best_scenario["scenario_name"],
        "strategic_value_index": best_scenario["strategic_value_index"],
        "description": f"{best_scenario['scenario_name']} achieved the highest Strategic Value Index of {best_scenario['strategic_value_index']:.2f}"
    })

    summary["key_findings"].append({
        "finding": "Worst Performing Scenario",
        "scenario": worst_scenario["scenario_name"],
        "strategic_value_index": worst_scenario["strategic_value_index"],
        "description": f"{worst_scenario['scenario_name']} achieved the lowest Strategic Value Index of {worst_scenario['strategic_value_index']:.2f}"
    })

    # Calculate performance spread
    performance_spread = best_scenario["strategic_value_index"] - worst_scenario["strategic_value_index"]
    summary["key_findings"].append({
        "finding": "Performance Spread",
        "value": performance_spread,
        "description": f"Performance difference between best and worst scenarios: {performance_spread:.2f}"
    })

    # Generate narrative summary
    narrative = f"""
    Multi-Scenario Analysis Summary:

    This analysis evaluated {len(scenario_results)} different scenarios using Pareto optimality and multi-criteria decision analysis.
    Each scenario was assessed across {len(comparison)} tasks, considering strategic value, time requirements, and cost factors.

    Key Results:
    • {best_scenario['scenario_name']} emerged as the top performer with a Strategic Value Index (SVI) of {best_scenario['strategic_value_index']:.2f}
    • {worst_scenario['scenario_name']} showed the lowest performance with an SVI of {worst_scenario['strategic_value_index']:.2f}
    • The performance gap between the best and worst scenarios is {performance_spread:.2f} SVI points

    Overall Performance Insights:
    """

    # Add scenario-specific insights
    for scenario in scenario_results:
        pareto_count = len(scenario["pareto_optimal_tasks"])
        total_tasks = pareto_count + len(scenario["non_pareto_tasks"])
        pareto_percentage = round((pareto_count / total_tasks) * 100, 1) if total_tasks > 0 else 0

        narrative += f"• {scenario['scenario_name']}: Achieved SVI of {scenario['strategic_value_index']:.2f} with {pareto_percentage}% Pareto optimal tasks\n"

    summary["narrative_summary"] = narrative.strip()

    # Generate scenario summaries
    for scenario in scenario_results:
        pareto_count = len(scenario["pareto_optimal_tasks"])
        non_pareto_count = len(scenario["non_pareto_tasks"])
        total_tasks = pareto_count + non_pareto_count

        summary["scenario_summaries"].append({
            "scenario_name": scenario["scenario_name"],
            "description": scenario["scenario_description"],
            "strategic_value_index": scenario["strategic_value_index"],
            "average_time": scenario["average_time"],
            "total_cost": scenario["total_cost"],
            "pareto_optimal_tasks": pareto_count,
            "non_pareto_tasks": non_pareto_count,
            "total_tasks": total_tasks,
            "pareto_percentage": round((pareto_count / total_tasks) * 100, 1) if total_tasks > 0 else 0,
            "summary": f"{scenario['scenario_name']}: SVI={scenario['strategic_value_index']:.2f}, Avg Time={scenario['average_time']:.2f}, Total Cost={scenario['total_cost']:.2f}, Pareto Tasks={pareto_count}/{total_tasks}"
        })

    return summary


def analyze_scenarios(scenarios_data):
    """
    Analyze multiple scenarios and return comprehensive results.
    Based on the full workflow from Sorting_Algo_phase3backup.py
    """
    scenario_results = []
    all_task_names = set()

    # Analyze each scenario
    for scenario in scenarios_data["scenarios"]:
        scenario_name = scenario["name"]
        scenario_description = scenario["description"]
        tasks = scenario["tasks"]
        weights = scenario["weights"]

        # Analyze this scenario
        pareto_results, non_pareto_results, min_score, max_score, avg_time, total_cost, strategic_index = analyze_strategy(tasks, weights)

        # Collect all task names for comparison
        for task in pareto_results + non_pareto_results:
            all_task_names.add(task["task_name"])

        # Store scenario result
        scenario_results.append({
            "scenario_name": scenario_name,
            "scenario_description": scenario_description,
            "pareto_optimal_tasks": pareto_results,
            "non_pareto_tasks": non_pareto_results,
            "min_possible_score": min_score,
            "max_possible_score": max_score,
            "average_time": avg_time,
            "total_cost": total_cost,
            "strategic_value_index": strategic_index
        })

        # Create simplified comparison matrix
    comparison = []
    for task_name in all_task_names:
        task_comparison = {
            "task_name": task_name,
            "scenario_performance": {}
        }

        for scenario_result in scenario_results:
            scenario_name = scenario_result["scenario_name"]

            # Find task in this scenario
            task_found = None
            for task in scenario_result["pareto_optimal_tasks"] + scenario_result["non_pareto_tasks"]:
                if task["task_name"] == task_name:
                    task_found = task
                    break

            if task_found:
                task_comparison["scenario_performance"][scenario_name] = {
                    "svi": task_found["svi"],
                    "is_pareto_optimal": task_found in scenario_result["pareto_optimal_tasks"]
                }
            else:
                task_comparison["scenario_performance"][scenario_name] = {
                    "svi": None,
                    "is_pareto_optimal": False
                }

        comparison.append(task_comparison)

    # Compute overall metrics
    overall_metrics = {
        "total_scenarios": len(scenario_results),
        "scenario_performance": []
    }

    for scenario_result in scenario_results:
        overall_metrics["scenario_performance"].append({
            "scenario_name": scenario_result["scenario_name"],
            "strategic_value_index": scenario_result["strategic_value_index"],
            "average_time": scenario_result["average_time"],
            "total_cost": scenario_result["total_cost"],
            "pareto_tasks_count": len(scenario_result["pareto_optimal_tasks"]),
            "non_pareto_tasks_count": len(scenario_result["non_pareto_tasks"])
        })

    # Generate summary
    summary = generate_summary(scenario_results, comparison)

    return {
        "summary": summary,
        "scenarios": scenario_results,
        "comparison": comparison,
        "overall_metrics": overall_metrics
    }


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

        score_1 = task_1["svi"] if task_1 else None
        score_2 = task_2["svi"] if task_2 else None

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

    # Generate narrative summary
    better_strategy = "Strategy 1" if strategy_1_results[6] > strategy_2_results[6] else "Strategy 2"
    performance_gap = abs(strategy_1_results[6] - strategy_2_results[6])

    narrative = f"""
    Strategy Comparison Analysis Summary:

    This analysis compared two strategic approaches using Pareto optimality and multi-criteria decision analysis.
    Each strategy was evaluated across {len(all_task_names)} tasks, considering strategic value, time requirements, and cost factors.

    Key Comparison Results:
    • {better_strategy} performed better with SVI of {max(strategy_1_results[6], strategy_2_results[6]):.2f}
    • Performance gap between strategies: {performance_gap:.2f} SVI points
    • Strategy 1: SVI={strategy_1_results[6]:.2f}, Avg Time={strategy_1_results[4]:.2f}, Total Cost={strategy_1_results[5]:.2f}
    • Strategy 2: SVI={strategy_2_results[6]:.2f}, Avg Time={strategy_2_results[4]:.2f}, Total Cost={strategy_2_results[5]:.2f}

    Pareto Optimality Comparison:
    • Strategy 1: {len(strategy_1_results[0])} Pareto optimal tasks out of {len(strategy_1_results[0]) + len(strategy_1_results[1])}
    • Strategy 2: {len(strategy_2_results[0])} Pareto optimal tasks out of {len(strategy_2_results[0]) + len(strategy_2_results[1])}

    Task-Level Insights:
    """

    # Add task-level insights
    for comp in comparison:
        if comp["score_difference"] != 0:
            if comp["score_difference"] > 0:
                narrative += f"• {comp['task_name']}: Strategy 2 performs better by {comp['score_difference']:.2f} SVI points\n"
            else:
                narrative += f"• {comp['task_name']}: Strategy 1 performs better by {abs(comp['score_difference']):.2f} SVI points\n"
        else:
            narrative += f"• {comp['task_name']}: Both strategies perform equally\n"

    narrative += f"""

    Recommendation:
    Based on the Strategic Value Index, {better_strategy} appears to be the more effective approach,
    achieving a {performance_gap:.2f} point advantage in overall strategic performance.
    """

    return {
        "narrative_summary": narrative.strip(),
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
