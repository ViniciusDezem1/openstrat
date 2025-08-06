# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import List, Dict
import os

from flask_openapi3 import Info, Tag
from flask_openapi3 import OpenAPI
from flask import jsonify

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


@app.post(
    "/fahp/calculate",
    summary="Calculate FAHP",
    tags=[fahp_tag],
    responses={"200": FAHPResponse},
)
def fahp_route(body: FAHPRequest):
    """
    Calculate FAHP endpoint.
    """
    result = calculate_fahp(body)
    return result.model_dump(mode="json")


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
