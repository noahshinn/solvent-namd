import yaml


with open('./temp.yaml') as f:
    d = yaml.load(f, Loader=yaml.Loader)
    print(d)
