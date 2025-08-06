# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import List, Dict

from flask_openapi3 import Info, Tag
from flask_openapi3 import OpenAPI

info = Info(title="ER API", version="1.0.0")
app = OpenAPI(__name__, info=info)

er_tag = Tag(name="er", description="Evidential Reasoning")


class BeliefDegree(BaseModel):
    grade: str
    degree: float


class Assessment(BaseModel):
    criterion: str
    belief_degrees: List[BeliefDegree]


class Alternative(BaseModel):
    name: str
    assessments: List[Assessment]


class ERRequest(BaseModel):
    alternatives: List[Alternative]
    grades: List[str]  # Ordered list of evaluation grades


def calculate_er(request: ERRequest):
    """
    Calculate Evidential Reasoning for decision making
    """
    # Here you would implement the actual ER calculation
    # This is a placeholder response
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "rankings": [
                {
                    "alternative": alt.name,
                    "belief_degrees": {
                        grade: sum(
                            bd.degree
                            for assessment in alt.assessments
                            for bd in assessment.belief_degrees
                            if bd.grade == grade
                        )
                        for grade in request.grades
                    }
                }
                for alt in request.alternatives
            ]
        }
    }


if __name__ == "__main__":
    app.run(debug=True)