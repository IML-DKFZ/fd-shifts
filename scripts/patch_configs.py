from pathlib import Path
from omegaconf import OmegaConf

base_path = Path("~/Experiments/fd-shifts_64/").expanduser()

for path in base_path.glob("**/config.yaml"):
    config = OmegaConf.load(path)
    if not hasattr(config.eval.query_studies, "new_class_study"):
        continue

    if "tinyimagenet" in config.eval.query_studies.new_class_study:
        print(config.eval.query_studies.new_class_study)
        index = config.eval.query_studies.new_class_study.index("tinyimagenet")
        config.eval.query_studies.new_class_study.pop(index)
        print(config.eval.query_studies.new_class_study)

        OmegaConf.save(config, path)
