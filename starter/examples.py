"""Example Census records used by the API docs, tests and request.py.

Both records are real rows from data/census.csv. The first is labelled
``>50K`` in the data, the second ``<=50K``. Keys use the original Census
feature names, including hyphens.
"""

HIGH_INCOME_EXAMPLE = {
    "age": 44,
    "workclass": "Private",
    "fnlgt": 198282,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}

LOW_INCOME_EXAMPLE = {
    "age": 18,
    "workclass": "Private",
    "fnlgt": 183930,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Other-service",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 12,
    "native-country": "United-States",
}

EXAMPLES = {
    "high_income": HIGH_INCOME_EXAMPLE,
    "low_income": LOW_INCOME_EXAMPLE,
}

EXPECTED_PREDICTIONS = {
    "high_income": ">50K",
    "low_income": "<=50K",
}
