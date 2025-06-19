import argparse
from sources.custom_datasets import arg_to_dataset

parser = argparse.ArgumentParser(
    prog="treat",
    description="Research project on hierarchical attention"
)

subparsers = parser.add_subparsers(description="The available actions with the project", dest="actions", required=True)

study_subparser = subparsers.add_parser("study")

study_subparser.add_argument("format", choices=["full", "bla"])
study_subparser.add_argument("--emb_model", default="sentence-transformers/all-MiniLM-L6-v2")
study_subparser.add_argument("--depth", default=7, choices=[2, 3, 4, 5, 6, 7, 8])
study_subparser.add_argument("--dataset", default="wikitext2", choices=[*arg_to_dataset.keys()])
study_subparser.add_argument("--batch-size", default=8, type=int)
study_subparser.add_argument("--display-freq", default=1, type=int)
study_subparser.add_argument("--save-plot-freq", default=50, type=int)
study_subparser.add_argument("--gating", default="summation", choices=["summation", "gated", "data-driven"])
study_subparser.add_argument("--scoring", default='sbs', choices=["sbs", "ivs"])
study_subparser.add_argument("--teacher-epochs", default=50, type=int)
study_subparser.add_argument("--student-epochs", default=10, type=int)


# emb_model: default: sentence-transformers/all-MiniLM-L6-v2
# depth: (2+), default: 7 (128)
# dataset: customs (wikitext2), any huggingface dataset
# batch_size: default 8
# display_freq: default (1)
# save_plot_frequency: default (50)
# gating: default: summation, gated, data-driven 
# scoring: default: sbs, ivs 

train_subparser = subparsers.add_parser("train")
