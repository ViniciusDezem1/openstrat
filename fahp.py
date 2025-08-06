# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np
from pyDecision.algorithm import fuzzy_ahp_method
from pydantic import BaseModel


class FuzzyMatrix(BaseModel):
    matrix: List[List[Tuple[float, float, float]]]


class FAHPRequest(BaseModel):
    integration: FuzzyMatrix = None
    costs: FuzzyMatrix = None
    relationship: FuzzyMatrix = None
    institutional: FuzzyMatrix = None
    performance: FuzzyMatrix = None
    customer: FuzzyMatrix = None
    marketplace: FuzzyMatrix = None

    model_config = dict(
        openapi_extra={
            "description": "Request body for FAHP calculation. Each matrix element is a tuple of (lower, middle, upper) values representing the fuzzy number.",
            "example": {
                "integration": {
                    "matrix": [
                        # Row 1: Comparing first criterion with others
                        [(1, 1, 1),     # Self-comparison (always 1,1,1)
                         (1/1, 1/2, 1/3)], # Comparing with second criterion
                        # Row 2: Comparing second criterion with others
                        [(1, 2, 3),     # Comparing with first criterion
                         (1, 1, 1)]      # Self-comparison
                    ]
                },
                "costs": {
                    "matrix": [
                        # Row 1: Comparing first criterion with others
                        [(1, 1, 1),     # Self-comparison
                         (1/1, 1/2, 1/3)], # Comparing with second criterion
                        # Row 2: Comparing second criterion with others
                        [(1, 2, 3),     # Comparing with first criterion
                         (1, 1, 1)]      # Self-comparison
                    ]
                },
                "relationship": {
                    "matrix": [
                        # Row 1: Comparing first criterion with others
                        [(1, 1, 1),     # Self-comparison
                         (2, 3, 4),     # Comparing with second criterion
                         (1/3, 1/4, 1/5)], # Comparing with third criterion
                        # Row 2: Comparing second criterion with others
                        [(1/2, 1/3, 1/4), # Comparing with first criterion
                         (1, 1, 1),     # Self-comparison
                         (1/4, 1/5, 1/6)], # Comparing with third criterion
                        # Row 3: Comparing third criterion with others
                        [(3, 4, 5),     # Comparing with first criterion
                         (4, 5, 6),     # Comparing with second criterion
                         (1, 1, 1)]      # Self-comparison
                    ]
                },
                "institutional": {
                    "matrix": [
                        # Row 1: Comparing first criterion with others
                        [(1, 1, 1),     # Self-comparison
                         (2, 3, 4)],    # Comparing with second criterion
                        # Row 2: Comparing second criterion with others
                        [(1/2, 1/3, 1/4), # Comparing with first criterion
                         (1, 1, 1)]      # Self-comparison
                    ]
                },
                "performance": {
                    "matrix": [
                        # Row 1: Comparing first criterion with others
                        [(1, 1, 1),     # Self-comparison
                         (2, 3, 4)],    # Comparing with second criterion
                        # Row 2: Comparing second criterion with others
                        [(1/2, 1/3, 1/4), # Comparing with first criterion
                         (1, 1, 1)]      # Self-comparison
                    ]
                },
                "customer": {
                    "matrix": [
                        # Row 1: Comparing first criterion with others
                        [(1, 1, 1),     # Self-comparison
                         (1, 2, 3),     # Comparing with second criterion
                         (2, 3, 4),     # Comparing with third criterion
                         (2, 3, 4),     # Comparing with fourth criterion
                         (1/2, 1/3, 1/4), # Comparing with fifth criterion
                         (2, 3, 4),     # Comparing with sixth criterion
                         (1/2, 1/3, 1/4), # Comparing with seventh criterion
                         (2, 3, 4)],    # Comparing with eighth criterion
                        # Row 2: Comparing second criterion with others
                        [(1/1, 1/2, 1/3), # Comparing with first criterion
                         (1, 1, 1),     # Self-comparison
                         (1/2, 1/3, 1/4), # Comparing with third criterion
                         (1/1, 1/2, 1/3), # Comparing with fourth criterion
                         (1/2, 1/3, 1/4), # Comparing with fifth criterion
                         (2, 3, 4),     # Comparing with sixth criterion
                         (1/2, 1/3, 1/4), # Comparing with seventh criterion
                         (2, 3, 4)],    # Comparing with eighth criterion
                        # Row 3: Comparing third criterion with others
                        [(1/2, 1/3, 1/4), # Comparing with first criterion
                         (2, 3, 4),     # Comparing with second criterion
                         (1, 1, 1),     # Self-comparison
                         (1/1, 1/2, 1/3), # Comparing with fourth criterion
                         (1/2, 1/3, 1/4), # Comparing with fifth criterion
                         (2, 3, 4),     # Comparing with sixth criterion
                         (1/3, 1/4, 1/5), # Comparing with seventh criterion
                         (2, 3, 4)],    # Comparing with eighth criterion
                        # Row 4: Comparing fourth criterion with others
                        [(1/2, 1/3, 1/4), # Comparing with first criterion
                         (1, 2, 3),     # Comparing with second criterion
                         (1, 2, 3),     # Comparing with third criterion
                         (1, 1, 1),     # Self-comparison
                         (1/2, 1/3, 1/4), # Comparing with fifth criterion
                         (2, 3, 4),     # Comparing with sixth criterion
                         (1/2, 1/3, 1/4), # Comparing with seventh criterion
                         (2, 3, 4)],    # Comparing with eighth criterion
                        # Row 5: Comparing fifth criterion with others
                        [(2, 3, 4),     # Comparing with first criterion
                         (2, 3, 4),     # Comparing with second criterion
                         (2, 3, 4),     # Comparing with third criterion
                         (2, 3, 4),     # Comparing with fourth criterion
                         (1, 1, 1),     # Self-comparison
                         (2, 3, 4),     # Comparing with sixth criterion
                         (1/1, 1/2, 1/3), # Comparing with seventh criterion
                         (3, 4, 5)],    # Comparing with eighth criterion
                        # Row 6: Comparing sixth criterion with others
                        [(1/2, 1/3, 1/4), # Comparing with first criterion
                         (1/2, 1/3, 1/4), # Comparing with second criterion
                         (1/2, 1/3, 1/4), # Comparing with third criterion
                         (1/2, 1/3, 1/4), # Comparing with fourth criterion
                         (1/2, 1/3, 1/4), # Comparing with fifth criterion
                         (1, 1, 1),     # Self-comparison
                         (1/2, 1/3, 1/4), # Comparing with seventh criterion
                         (2, 3, 4)],    # Comparing with eighth criterion
                        # Row 7: Comparing seventh criterion with others
                        [(1, 2, 3),     # Comparing with first criterion
                         (2, 3, 4),     # Comparing with second criterion
                         (3, 4, 5),     # Comparing with third criterion
                         (2, 3, 4),     # Comparing with fourth criterion
                         (1, 2, 3),     # Comparing with fifth criterion
                         (2, 3, 4),     # Comparing with sixth criterion
                         (1, 1, 1),     # Self-comparison
                         (3, 4, 5)],    # Comparing with eighth criterion
                        # Row 8: Comparing eighth criterion with others
                        [(1/2, 1/3, 1/4), # Comparing with first criterion
                         (1/2, 1/3, 1/4), # Comparing with second criterion
                         (1/2, 1/3, 1/4), # Comparing with third criterion
                         (1/2, 1/3, 1/4), # Comparing with fourth criterion
                         (1/3, 1/4, 1/5), # Comparing with fifth criterion
                         (1/3, 1/4, 1/5), # Comparing with sixth criterion
                         (1/3, 1/4, 1/5), # Comparing with seventh criterion
                         (1, 1, 1)]      # Self-comparison
                    ]
                },
                "marketplace": {
                    "matrix": [
                        # Row 1: Comparing first criterion with others
                        [(1, 1, 1),     # Self-comparison
                         (1/2, 1/3, 1/4), # Comparing with second criterion
                         (1/3, 1/4, 1/5)], # Comparing with third criterion
                        # Row 2: Comparing second criterion with others
                        [(2, 3, 4),     # Comparing with first criterion
                         (1, 1, 1),     # Self-comparison
                         (1/3, 1/4, 1/5)], # Comparing with third criterion
                        # Row 3: Comparing third criterion with others
                        [(3, 4, 5),     # Comparing with first criterion
                         (3, 4, 5),     # Comparing with second criterion
                         (1, 1, 1)]      # Self-comparison
                    ]
                }
            }
        }
    )


class FAHPResponse(BaseModel):
    code: int
    message: str
    data: List[dict]


def calculate_fahp(request: FAHPRequest) -> FAHPResponse:
    """
    Calculate FAHP (Fuzzy Analytic Hierarchy Process) for decision making
    """
    results = []

    judgments = {
        "integration judgment": request.integration.matrix if request.integration else None,
        "costs x flexible solutions judgment": request.costs.matrix if request.costs else None,
        "relationship subdimensions judgment": request.relationship.matrix if request.relationship else None,
        "institutional image judgment": request.institutional.matrix if request.institutional else None,
        "performance judgment": request.performance.matrix if request.performance else None,
        "customer at the centre judgment": request.customer.matrix if request.customer else None,
        "marketplace judgment": request.marketplace.matrix if request.marketplace else None
    }

    for name, matrix in judgments.items():
        if matrix is not None:
            # Call Fuzzy AHP Function
            fuzzy_weights, defuzzified_weights, normalized_weights, rc = fuzzy_ahp_method(matrix)

            # Format results exactly like the original file
            result = {
                "name": name,
                "fuzzy_weights": {
                    f"g{i+1}": [float(w[0]), float(w[1]), float(w[2])]
                    for i, w in enumerate(fuzzy_weights)
                },
                "defuzzified_weights": {
                    f"g{i+1}": float(w)
                    for i, w in enumerate(defuzzified_weights)
                },
                "normalized_weights": {
                    f"g{i+1}": float(w)
                    for i, w in enumerate(normalized_weights)
                },
                "consistency_ratio": float(rc),
                "is_consistent": str(rc <= 0.10).lower(),
                "consistency_message": "The solution is inconsistent, the pairwise comparisons must be reviewed" if rc > 0.10 else "The solution is consistent"
            }
            results.append(result)

    return FAHPResponse(code=0, message="ok", data=results)