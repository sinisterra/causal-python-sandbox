import pandas as pd

from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest

df = pd.read_csv("lucas.csv")

variable_types = {}
for col in df.columns:
    variable_types[col] = "b"


# # run the search
ic_algorithm = IC(RobustRegressionTest)
print(ic_algorithm)
# graph = ic_algorithm.search(df, variable_types)
