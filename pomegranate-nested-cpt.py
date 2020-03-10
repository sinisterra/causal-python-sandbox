import pandas as pd
from pomegranate import (
    DiscreteDistribution,
    ConditionalProbabilityTable,
    Node,
    BayesianNetwork,
)

df = pd.read_csv("./datasets/lucas.csv")


def buildCpt(target, parents):
    cols = [*parents, target]
    # assuming at least one column remains...
    s = (df.groupby(cols).count().iloc[:, 0].rename("prob") / len(df)).to_frame()
    t = s.unstack(fill_value=0).stack().reset_index()
    print(t)
    return t.values.tolist()


def singleVariable(target):
    oneValue = df[target].value_counts()[1] / len(df) or 0
    return {0: 1 - oneValue, 1: oneValue}


anxiety = DiscreteDistribution(singleVariable("Anxiety"))
peer_pressure = DiscreteDistribution(singleVariable("Peer_Pressure"))
genetics = DiscreteDistribution(singleVariable("Genetics"))
allergy = DiscreteDistribution(singleVariable("Allergy"))


smoking = ConditionalProbabilityTable(
    buildCpt("Smoking", ["Anxiety", "Peer_Pressure"]), [anxiety, peer_pressure]
)
lung_cancer = ConditionalProbabilityTable(
    buildCpt("Lung_cancer", ["Smoking", "Genetics"]), [smoking, genetics]
)
yellow_fingers = ConditionalProbabilityTable(
    buildCpt("Yellow_Fingers", ["Smoking"]), [smoking]
)
attention_disorder = ConditionalProbabilityTable(
    buildCpt("Attention_Disorder", ["Genetics"]), [genetics]
)
coughing = ConditionalProbabilityTable(
    buildCpt("Coughing", ["Allergy", "Lung_cancer"]), [allergy, lung_cancer]
)
fatigue = ConditionalProbabilityTable(
    buildCpt("Fatigue", ["Lung_cancer", "Coughing"]), [lung_cancer, coughing]
)
car_accident = ConditionalProbabilityTable(
    buildCpt("Car_Accident", ["Fatigue", "Attention_Disorder"]),
    [fatigue, attention_disorder],
)

sAnxiety = Node(anxiety, name="Anxiety")
sPeerPressure = Node(peer_pressure, name="Peer_Pressure")
sSmoking = Node(smoking, name="Smoking")
sGenetics = Node(genetics, name="Genetics")
sLungCancer = Node(lung_cancer, name="Lung_cancer")
sYellowFingers = Node(yellow_fingers, name="Yellow_Fingers")
sAttentionDisorder = Node(attention_disorder, name="Attention_Disorder")
sCoughing = Node(coughing, name="Coughing")
sFatigue = Node(fatigue, name="Fatigue")
sCarAccident = Node(car_accident, name="Car_Accident")
sAllergy = Node(allergy, name="Allergy")


model = BayesianNetwork("Smoking Risk")
model.add_nodes(
    sAnxiety,
    sPeerPressure,
    sSmoking,
    sGenetics,
    sLungCancer,
    sAllergy,
    sCoughing,
    sFatigue,
    sAttentionDisorder,
    sCarAccident,
)
model.add_edge(sAnxiety, sSmoking)
model.add_edge(sPeerPressure, sSmoking)
model.add_edge(sSmoking, sLungCancer)
model.add_edge(sGenetics, sLungCancer)
model.add_edge(sGenetics, sAttentionDisorder)
model.add_edge(sAllergy, sCoughing)
model.add_edge(sLungCancer, sCoughing)
model.add_edge(sCoughing, sFatigue)
model.add_edge(sLungCancer, sFatigue)
model.add_edge(sFatigue, sCarAccident)
model.add_edge(sAttentionDisorder, sCarAccident)
model.bake()

prediction = model.predict_proba({})
states = [
    "Anxiety",
    "Peer_pressure",
    "Smoking",
    "Genetics",
    "Lung_cancer",
    "Allergy",
    "Coughing",
    "Fatigue",
    "Attention_Disorder",
    "Car_Accident",
]

print(prediction)
# print([(s, p) for (s, p) in list(zip(states, prediction))])

