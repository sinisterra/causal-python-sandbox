import cdt
import pandas as pd
import networkx as nx


dataset_name = "intervention_xyz"
df = pd.read_csv(f"./datasets/{dataset_name}.csv")

# Produce a graph skeleton to pre-compute the structure
# skeleton = cdt.independence.graph.Glasso().predict(df)
# nx.drawing.nx_agraph.write_dot(skeleton, "./skeleton.dot")

algorithms = ["CCDr", "GES", "GIES", "LiNGAM", "PC"]


models = {
    "CCDr": cdt.causality.graph.CCDr,
    "GES": cdt.causality.graph.GES,
    "GIES": cdt.causality.graph.GIES,
    "LiNGAM": cdt.causality.graph.LiNGAM,
    "PC": cdt.causality.graph.PC,
    # "SAMv1": cdt.causality.graph.SAMv1, # takes too long
    # "CAM": cdt.causality.graph.CAM,  # failing
    # "CGNN": cdt.causality.graph.CGNN,  # takes a lot of time (super-exponential)
    # "SAM": cdt.causality.graph.SAM,  # lots of iterations (10,000)
}

# try some models sequentially, skip those that fail
for model_name, instantiate_model in models.items():
    try:
        print(f"Trying out {model_name}...")
        model = instantiate_model()
        output_graph = model.predict(df)
        nx.drawing.nx_agraph.write_dot(
            output_graph, f"./models/{dataset_name}_{model_name}.dot"
        )
        print(f"Model {model_name} finished.")
    except Exception as e:
        print(e)
        print(f"{model_name} failed.")
