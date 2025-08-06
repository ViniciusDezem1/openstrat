# Open Finance Decision Support API

A Flask-based API that implements various Multi-Criteria Decision Analysis (MCDA) methods to support financial decision-making processes.

## Features

- **FAHP (Fuzzy Analytic Hierarchy Process)**: Implemented and ready to use
- **MCDA (Multi-Criteria Decision Analysis)**: Work in Progress
- **ER (Evidential Reasoning)**: Work in Progress

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ViniciusDezem1/openfinancedecision.git
cd openfinancedecision
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

The application can be run in two ways:

1. Using Flask CLI (recommended):
```bash
flask --app app run
```

The API will be available at `http://127.0.0.1:5000/openapi/scalar`

### Environment Variables

You can configure the application using environment variables:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development  # For development mode
export FLASK_DEBUG=1         # Enable debug mode
```

## API Documentation

Once the server is running, you can access the OpenAPI documentation at:
- Swagger UI: `http://127.0.0.1:5000/openapi/scalar`
- ReDoc: `http://127.0.0.1:5000/openapi/scalar`

### Available Endpoints

#### FAHP (Fuzzy Analytic Hierarchy Process)
- **Endpoint**: `/fahp/calculate`
- **Method**: POST
- **Description**: Calculates weights using the Fuzzy Analytic Hierarchy Process method
- **Request Body**:
  ```json
  {
    "criteria": ["criterion1", "criterion2", ...],
    "fuzzy_matrix": [
      [1, 1, 1, 1, 1, 1],
      [1/3, 1/2, 1, 1, 1, 1],
      ...
    ]
  }
  ```
- **Response**:
  ```json
  {
    "weights": [0.25, 0.35, ...],
    "consistency_ratio": 0.05
  }
  ```

#### MCDA (Multi-Criteria Decision Analysis)
- **Status**: Work in Progress
- **Endpoint**: `/mcda/calculate`
- **Method**: POST

#### ER (Evidential Reasoning)
- **Status**: Work in Progress
- **Endpoint**: `/er/calculate`
- **Method**: POST

## Development

### Project Structure
```
openfinancedecision/
├── app.py              # Main Flask application
├── fahp.py            # FAHP implementation
├── mcda.py            # MCDA implementation (WIP)
├── er.py              # ER implementation (WIP)
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

### Adding New Methods

To add a new MCDA method:
1. Create a new Python file for the method implementation
2. Define the request and response models using Pydantic
3. Implement the calculation logic
4. Add the route to `app.py`
5. Update the OpenAPI documentation

