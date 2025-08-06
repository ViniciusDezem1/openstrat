# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import List, Dict, Literal

from flask_openapi3 import Info, Tag
from flask_openapi3 import OpenAPI

info = Info(title="MCDA API", version="1.0.0")
app = OpenAPI(__name__, info=info)

mcda_tag = Tag(name="mcda", description="Multi-Criteria Decision Analysis")


class Criterion(BaseModel):
    name: str
    weight: float
    direction: Literal["maximize", "minimize"]


class Alternative(BaseModel):
    name: str
    values: Dict[str, float]


class MCDARequest(BaseModel):
    criteria: List[Criterion]
    alternatives: List[Alternative]
    method: Literal["topsis", "promethee", "electre"]


def calculate_mcda(request: MCDARequest):
    """
    Calculate MCDA using various methods (TOPSIS, PROMETHEE, or ELECTRE)
    """
    # Here you would implement the actual MCDA calculation based on the method
    # This is a placeholder response
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "method": request.method,
            "rankings": [
                {"alternative": alt.name, "score": sum(alt.values.values())}
                for alt in request.alternatives
            ]
        }
    }


if __name__ == "__main__":
    app.run(debug=True)