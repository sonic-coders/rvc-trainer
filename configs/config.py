import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--colab", action="store_true", help="Launch in colab")
args = parser.parse_args()
iscolab = args.colab
