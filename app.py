import os
from typing import Any, List, Optional, Tuple

import numpy as np
from flask import jsonify, redirect
from flask_openapi3 import Info, OpenAPI, Tag
from pydantic import BaseModel, Field
from pyDecision.algorithm import fuzzy_ahp_method

from sorting import (
    SortingRequest, SortingResponse, ScenariosRequest, ScenariosResponse,
    StrategyComparisonRequest, StrategyComparisonResponse,
    analyze_strategy, analyze_scenarios, compare_strategies, generate_single_strategy_summary
)

info = Info(title="MCDA Decision Support API", version="1.0.0")
app = OpenAPI(__name__, info=info)

# Define tags
fahp_tag = Tag(name="fahp", description="Fuzzy Analytic Hierarchy Process")
sorting_tag = Tag(name="sorting", description="Pareto Optimality and Strategy Comparison")


@app.get("/", summary="Health check", tags=[])
def root():
    """
    Redirects to the OpenAPI documentation.
    """
    return redirect("/openapi/scalar")


@app.get("/open-finance", summary="OpenAPI Spec", tags=[])
def open_finance():
    """
    Returns the OpenAPI specification as JSON.
    """
    return jsonify(app.openapi())


fahp_example = {
    "matrix": [
        [[1, 1, 1], [0.5, 0.333, 0.25], [0.5, 0.333, 0.25], [
            0.5, 0.333, 0.25], [2, 3, 4], [2, 3, 4], [0.5, 0.333, 0.25]],
        [[2, 3, 4], [1, 1, 1], [0.5, 0.333, 0.25], [
            0.5, 0.333, 0.25], [2, 3, 4], [2, 3, 4], [2, 3, 4]],
        [[2, 3, 4], [2, 3, 4], [1, 1, 1], [1, 2, 3], [
            2, 3, 4], [2, 3, 4], [0.5, 0.333, 0.25]],
        [[2, 3, 4], [2, 3, 4], [1, 0.5, 0.333], [
            1, 1, 1], [2, 3, 4], [2, 3, 4], [1, 1, 0.5]],
        [[0.5, 0.333, 0.25], [0.5, 0.333, 0.25], [0.5, 0.333, 0.25], [
            0.5, 0.333, 0.25], [1, 1, 1], [1, 2, 3], [0.5, 0.333, 0.25]],
        [[0.5, 0.333, 0.25], [0.5, 0.333, 0.25], [0.5, 0.333, 0.25], [
            0.5, 0.333, 0.25], [1, 0.5, 0.333], [1, 1, 1], [0.5, 0.333, 0.25]],
        [[2, 3, 4], [0.5, 0.333, 0.25], [2, 3, 4], [
            1, 1, 2], [2, 3, 4], [2, 3, 4], [1, 1, 1]]
    ],
    "criteria_names": [
        "g1", "g2", "g3", "g4", "g5", "g6", "g7"
    ]
}


class FAHPMatrixRequest(BaseModel):
    matrix: List[List[Any]] = Field(
        default=fahp_example["matrix"],
        description="Fuzzy comparison matrix where each element is [lower, middle, upper]"
    )
    criteria_names: Optional[List[str]] = Field(
        default=fahp_example["criteria_names"],
        description="Names of the criteria being compared"
    )


class FAHPResponse(BaseModel):
    fuzzy_weights: List[List[float]]
    defuzzified_weights: List[float]
    normalized_weights: List[float]
    consistency_ratio: float
    consistency: bool


@app.post(
    "/fahp/calculate",
    summary="Calculate FAHP",
    tags=[fahp_tag],
    responses={"200": FAHPResponse}
)
def fahp_route(body: FAHPMatrixRequest):
    """
    Calculate FAHP endpoint.
    Uses the matrix from the request body.

    Example request body:
    ```json
    {
      "matrix": [
        [ [1, 1, 1], [0.5, 0.333, 0.25], ... ],
      ],
      "criteria_names": null
    }
    ```

    Returns verbose results for each group, matching example.py logic.
    """
    dataset = body.matrix

    # Defensive: If matrix is missing, return error
    if not dataset:
        return jsonify({"error": "Matrix is required"}), 400

    # Calculation (same as example.py)
    fuzzy_weights, defuzzified_weights, normalized_weights, rc = fuzzy_ahp_method(
        dataset)

    verbose = {}

    # Fuzzy weights per group
    verbose["fuzzy_weights"] = []
    for i, fw in enumerate(fuzzy_weights):
        verbose["fuzzy_weights"].append({
            "group": f"g{i+1}",
            "weights": list(np.around(fw, 3))
        })

    # Defuzzified weights per group
    verbose["defuzzified_weights"] = []
    for i, dw in enumerate(defuzzified_weights):
        verbose["defuzzified_weights"].append({
            "group": f"g{i+1}",
            "weight": round(dw, 3)
        })

    # Normalized weights per group
    verbose["normalized_weights"] = []
    for i, nw in enumerate(normalized_weights):
        verbose["normalized_weights"].append({
            "group": f"g{i+1}",
            "weight": round(nw, 3)
        })

    # Consistency ratio and message
    threshold = 0.999
    verbose["consistency_ratio"] = round(rc, 2)
    verbose["consistency"] = bool(rc <= threshold)
    verbose["consistency_message"] = (
        "Consistent" if rc <= threshold else "Inconsistent, the pairwise comparisons must be reviewed"
    )

    # Generate narrative summary
    total_criteria = len(dataset)
    best_criterion = max(verbose["normalized_weights"], key=lambda x: x["weight"])
    worst_criterion = min(verbose["normalized_weights"], key=lambda x: x["weight"])

    narrative = f"""
    Fuzzy Analytic Hierarchy Process (FAHP) Analysis Summary:

    This analysis evaluated {total_criteria} criteria using fuzzy pairwise comparisons to determine their relative importance weights.
    The FAHP method incorporates uncertainty in decision-making through triangular fuzzy numbers.

    Key Results:
    • Total Criteria Analyzed: {total_criteria}
    • Most Important Criterion: {best_criterion['group']} (Weight: {best_criterion['weight']:.3f})
    • Least Important Criterion: {worst_criterion['group']} (Weight: {worst_criterion['weight']:.3f})
    • Consistency Ratio: {verbose['consistency_ratio']:.2f} {'(Consistent)' if verbose['consistency'] else '(Inconsistent)'}

    Weight Distribution:
    """

    # Add weight distribution
    for weight_info in verbose["normalized_weights"]:
        narrative += f"• {weight_info['group']}: {weight_info['weight']:.3f} ({weight_info['weight']*100:.1f}%)\n"

    narrative += f"""

    Analysis Quality:
    • Consistency Status: {'✓ Consistent - Results are reliable' if verbose['consistency'] else '✗ Inconsistent - Review pairwise comparisons'}
    • Fuzzy Weights: Represent uncertainty ranges for each criterion
    • Defuzzified Weights: Single values for practical decision-making
    • Normalized Weights: Sum to 1.0 for relative importance assessment
    """

    verbose["narrative_summary"] = narrative.strip()

    return jsonify(verbose)


@app.post(
    "/sorting/calculate",
    summary="Calculate Sorting",
    tags=[sorting_tag],
    responses={"200": SortingResponse}
)
def sorting_route(body: SortingRequest):
    """
    Calculate Sorting endpoint.
    Uses the tasks and weights from the request body.

    Example request body:

    ```json
    {
      "tasks": [
        {"name": "Automated Robo-Advisory", "score": 1.94, "time": 6.0, "cost": 5.0},
        {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
        ...
      ],
      "weights": {"w_time": 1.05, "w_cost": 1.05}
    }
    ```

    Returns verbose results for Pareto optimality analysis, matching Sorting_Algo_phase3backup.py logic.
    """
    dataset = body.tasks
    weights = body.weights

    # Defensive: If tasks are missing, return error
    if not dataset:
        return jsonify({"error": "Tasks are required"}), 400

    # Calculation (same as Sorting_Algo_phase3backup.py)
    pareto_results, non_pareto_results, min_score, max_score, avg_time, total_cost, strategic_index = analyze_strategy(
        dataset, weights)

    # Generate summary
    summary = generate_single_strategy_summary(pareto_results, non_pareto_results, min_score, max_score, avg_time, total_cost, strategic_index)

    verbose = {}

    # Summary
    verbose["summary"] = summary

    # Pareto optimal tasks
    verbose["pareto_optimal_tasks"] = pareto_results

    # Non-Pareto tasks
    verbose["non_pareto_tasks"] = non_pareto_results

    # Metrics
    verbose["min_possible_score"] = round(min_score, 2)
    verbose["max_possible_score"] = round(max_score, 2)
    verbose["average_time"] = round(avg_time, 2)
    verbose["total_cost"] = round(total_cost, 2)
    verbose["strategic_value_index"] = round(strategic_index, 2)

    return jsonify(verbose)


@app.post(
    "/sorting/compare",
    summary="Compare Multiple Strategies",
    tags=[sorting_tag],
    responses={"200": StrategyComparisonResponse}
)
def sorting_compare_route(body: StrategyComparisonRequest):
    """
    Compare multiple strategies using Pareto optimality and multi-criteria decision analysis.
    Based on the full workflow from Sorting_Algo_phase3backup.py

    Example request body:

    ```json
    {
      "strategy_1_dataset_1": {
        "tasks": [
          {"name": "Automated Robo-Advisory", "score": 1.94, "time": 6.0, "cost": 5.0},
          {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
          ...
        ],
        "weights": {"w_time": 1.05, "w_cost": 1.05}
      },
      "strategy_2_dataset_2": {
        "tasks": [
          {"name": "Automated Robo-Advisory", "score": 1.94, "time": 4.0, "cost": 4.0},
          {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
          ...
        ],
        "weights": {"w_time": 1.0, "w_cost": 1.0}
      }
    }
    ```

    Returns comprehensive comparison including:
    - Individual strategy analysis (Pareto optimal tasks, metrics)
    - Side-by-side task comparison with score differences
    - Overall strategy metrics comparison
    """
    strategy_1_data = body.strategy_1_dataset_1
    strategy_2_data = body.strategy_2_dataset_2

    # Defensive: If data is missing, return error
    if not strategy_1_data or not strategy_2_data:
        return jsonify({"error": "Both strategy datasets are required"}), 400

    # Calculate comparison
    result = compare_strategies(strategy_1_data, strategy_2_data)

    return jsonify(result)


@app.post(
    "/sorting/scenarios",
    summary="Analyze Multiple Scenarios",
    tags=[sorting_tag],
    responses={"200": ScenariosResponse}
)
def sorting_scenarios_route(body: ScenariosRequest):
    """
    Analyze multiple scenarios using Pareto optimality and multi-criteria decision analysis.
    Based on the full workflow from Sorting_Algo_phase3backup.py

    Example request body:
    ```json
    {
      "scenarios": [
        {
          "name": "Strategy 1 - Dataset 1",
          "description": "In-house approach with original data",
          "tasks": [
            {"name": "Automated Robo-Advisory", "score": 1.94, "time": 6.0, "cost": 5.0},
            {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
            ...
          ],
          "weights": {"w_time": 1.05, "w_cost": 1.05}
        },
        {
          "name": "Strategy 2 - Dataset 2",
          "description": "Outsource approach with modified data",
          "tasks": [
            {"name": "Automated Robo-Advisory", "score": 1.94, "time": 4.0, "cost": 4.0},
            {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0},
            ...
          ],
          "weights": {"w_time": 1.0, "w_cost": 1.0}
        }
      ]
    }
    ```

    Returns comprehensive analysis including:
    - Individual scenario results (Pareto optimal tasks, metrics)
    - Cross-scenario comparison matrix
    - Overall performance metrics for all scenarios
    """
    scenarios_data = body.scenarios

    # Defensive: If scenarios are missing, return error
    if not scenarios_data:
        return jsonify({"error": "Scenarios are required"}), 400

    # Calculate scenarios analysis
    result = analyze_scenarios({"scenarios": scenarios_data})

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)
