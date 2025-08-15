# MCDA Decision Support API

A Flask-based API for Multi-Criteria Decision Analysis (MCDA) with a focus on Fuzzy Analytic Hierarchy Process (FAHP) and Pareto optimality analysis. The API provides endpoints for analyzing criteria weights, comparing strategies, and scenario analysis using multi-criteria data, with comprehensive narrative summaries and interactive OpenAPI (Scalar) documentation UI.

## Features

- **Fuzzy Analytic Hierarchy Process (FAHP)**: Determine criteria importance weights using fuzzy pairwise comparisons with consistency validation.
- **Pareto Optimality Analysis**: Identify Pareto optimal and non-optimal tasks based on strategic score, time, and cost.
- **Strategy Comparison**: Compare two strategies side-by-side, including detailed task-level and overall metrics.
- **Multi-Scenario Analysis**: Analyze multiple scenarios, generate comprehensive summaries, and compare performance across scenarios.
- **Narrative Summaries**: Rich textual explanations of analysis results for better decision-making insights.
- **OpenAPI Documentation**: Interactive Scalar UI available at `/openapi/scalar`.

## API Endpoints

### FAHP Analysis
- `POST /fahp/calculate`: Calculate FAHP weights for criteria using fuzzy pairwise comparisons.

### Strategy Analysis & Comparison
- `POST /sorting/calculate`: Analyze a single strategy and return Pareto optimality results with narrative summary.
- `POST /sorting/compare`: Compare two strategies and return detailed comparison results with recommendations.
- `POST /sorting/scenarios`: Analyze multiple scenarios and return comprehensive summaries and comparison matrices.

### Documentation
- `GET /openapi/scalar`: Access the Scalar OpenAPI documentation UI.
- **Production URL**: [https://openstrat-production.up.railway.app/openapi/scalar](https://openstrat-production.up.railway.app/openapi/scalar)

## Example Requests

### FAHP Analysis
```json
POST /fahp/calculate
{
  "matrix": [
    [[1, 1, 1], [0.5, 0.333, 0.25], [0.5, 0.333, 0.25]],
    [[2, 3, 4], [1, 1, 1], [0.5, 0.333, 0.25]],
    [[2, 3, 4], [2, 3, 4], [1, 1, 1]]
  ],
  "criteria_names": ["g1", "g2", "g3"]
}
```

### Strategy Analysis
```json
POST /sorting/calculate
{
  "tasks": [
    {"name": "Automated Robo-Advisory", "score": 1.94, "time": 6.0, "cost": 5.0},
    {"name": "Product Visibility", "score": 2.3, "time": 3.0, "cost": 2.0}
  ],
  "weights": {"w_time": 1.05, "w_cost": 1.05}
}
```

### Strategy Comparison
```json
POST /sorting/compare
{
  "strategy_1_dataset_1": {
    "tasks": [
      {"name": "Task A", "score": 1.94, "time": 6.0, "cost": 5.0}
    ],
    "weights": {"w_time": 1.05, "w_cost": 1.05}
  },
  "strategy_2_dataset_2": {
    "tasks": [
      {"name": "Task A", "score": 1.94, "time": 4.0, "cost": 4.0}
    ],
    "weights": {"w_time": 1.0, "w_cost": 1.0}
  }
}
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yourrepo.git
    cd yourrepo
    ```

2. (Recommended) Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

You can run the API locally with:

```sh
flask --app app run
```

Or, using Gunicorn (recommended for production):

```sh
gunicorn app:app
```

The Scalar UI will be available at [http://127.0.0.1:5000/openapi/scalar](http://127.0.0.1:5000/openapi/scalar).

**Production API**: [https://openstrat-production.up.railway.app/openapi/scalar](https://openstrat-production.up.railway.app/openapi/scalar)

## Deployment

To deploy using Nixpacks or similar platforms, ensure your [nixpacks.toml](nixpacks.toml) contains:

```toml
[start]
cmd = "gunicorn app:app"
```

Set the `PORT` environment variable if needed (defaults to 3000).

---

For more details on the API request/response models, see the Scalar UI or review the source code.

