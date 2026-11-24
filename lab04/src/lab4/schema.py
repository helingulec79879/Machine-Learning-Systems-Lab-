"""Schema / data dictionary for UCI Adult (from adult.names)."""

INT_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "native-country"]
TARGET = "income"
FEATURES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]
ALL_COLS = FEATURES + [TARGET]

TYPES = {c: "int" for c in INT_COLS}
TYPES.update({c: "category" for c in CAT_COLS})
TYPES[TARGET] = "category"

# Ranges required by the lab sheet (inclusive)
RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

# Extra data-dictionary ranges (reported as informational checks)
EXTRA_RANGES = {
    "education-num": (1, 16),
    "fnlwgt": (1, None),
}

MISSING_TOKEN = "?"

ALLOWED = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                  "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
                  "Doctorate", "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                       "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                       "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                       "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
                       "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    "income": ["<=50K", ">50K"],
}
# Null policy: '?' is an allowed category (it means "missing")
for _c in CAT_COLS:
    ALLOWED[_c] = ALLOWED[_c] + [MISSING_TOKEN]

# education <-> education-num must agree (data dictionary consistency)
EDU_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
           "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
           "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}
