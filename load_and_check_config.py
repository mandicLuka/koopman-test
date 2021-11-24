import os, yaml
def load_and_check_config(name:str) -> dict:
    with open(os.path.join("config", f"{name}.yml"), "r") as stream:
        #try:
            cfg = yaml.safe_load(stream)
        #except yaml.YAMLError as exc:
        #    print(exc)

    return cfg