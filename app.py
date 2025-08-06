# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import List, Tuple, Dict, Optional, Any
import os

from flask_openapi3 import Info, Tag
from flask_openapi3 import OpenAPI
from flask import jsonify
from pyDecision.algorithm import fuzzy_ahp_method
import numpy as np

# Import the calculation functions and schemas from other files
from fahp import calculate_fahp, FAHPRequest, FAHPResponse
from mcda import calculate_mcda, MCDARequest
from er import calculate_er, ERRequest

info = Info(title="MCDA Decision Support API", version="1.0.0")
app = OpenAPI(__name__, info=info)

# Define tags
fahp_tag = Tag(name="fahp", description="Fuzzy Analytic Hierarchy Process")
mcda_tag = Tag(name="mcda", description="Multi-Criteria Decision Analysis")
er_tag = Tag(name="er", description="Evidential Reasoning")


@app.get("/", summary="Health check", tags=[])
def root():
    """
    Health check endpoint that returns a scalar string.
    """
    return "OK"


@app.get("/open-finance", summary="OpenAPI Spec", tags=[])
def open_finance():
    """
    Returns the OpenAPI specification as JSON.
    """
    return jsonify(app.openapi())


class FAHPMatrixRequest(BaseModel):
    matrix: Optional[List[List[Any]]] = None
    criteria_names: Optional[List[str]] = None


class FAHPResponse(BaseModel):
    fuzzy_weights: List[List[float]]
    defuzzified_weights: List[float]
    normalized_weights: List[float]
    consistency_ratio: float
    consistency: bool


@app.post(
    "/fahp/calculate",
    summary="Calculate FAHP (example dataset, ignores input)",
    tags=[fahp_tag],
    responses={"200": FAHPResponse}
)
def fahp_route():
    """
    Calculate FAHP endpoint.
    Ignores input and always calculates using the example.py dataset.
    Returns verbose results for each group, matching example.py output.
    """
    # Hardcoded example dataset from example.py
    dataset = [
        [ (1, 1, 1), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (2, 3, 4), (2, 3, 4), (1/2, 1/3, 1/4) ],
        [ (2, 3, 4), (1, 1, 1), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (2, 3, 4), (2, 3, 4), (2, 3, 4) ],
        [ (2, 3, 4), (2, 3, 4), (1, 1, 1), (1, 2, 3), (2, 3, 4), (2, 3, 4), (1/2, 1/3, 1/4) ],
        [ (2, 3, 4), (2, 3, 4), (1, 1/2, 1/3), (1, 1, 1), (2, 3, 4), (2, 3, 4), (1, 1, 1/2) ],
        [ (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1, 1, 1), (1, 2, 3), (1/2, 1/3, 1/4) ],
        [ (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4), (1, 1/2, 1/3), (1, 1, 1), (1/2, 1/3, 1/4) ],
        [ (2, 3, 4), (1/2, 1/3, 1/4), (2, 3, 4), (1, 1, 2), (2, 3, 4), (2, 3, 4), (1, 1, 1) ]
    ]

    fuzzy_weights, defuzzified_weights, normalized_weights, rc = fuzzy_ahp_method(dataset)

    # Prepare verbose output
    verbose = {}

    # Fuzzy weights per group
    verbose["fuzzy_weights"] = []
    for i, fw in enumerate(fuzzy_weights):
        verbose["fuzzy_weights"].append({
            "group": f"g{i+1}",
            "weights": list(np.around(fw, 3))
        })

    # Defuzzified (crisp) weights per group
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

    # Consistency ratio and status
    verbose["consistency_ratio"] = round(rc, 2)
    verbose["consistency"] = bool(rc <= 0.10)  # Explicitly cast to bool
    verbose["consistency_message"] = (
        "Consistent" if rc <= 0.10 else "Inconsistent, the pairwise comparisons must be reviewed"
    )

    return jsonify(verbose)


@app.post("/mcda/calculate", summary="Calculate MCDA", tags=[mcda_tag])
def mcda_route(request: MCDARequest):
    """
    Calculate MCDA endpoint.
    """
    return calculate_mcda(request)


@app.post("/er/calculate", summary="Calculate ER", tags=[er_tag])
def er_route(request: ERRequest):
    """
    Calculate ER endpoint.
    """
    return calculate_er(request)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)
