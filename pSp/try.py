import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='transfer facial expression for videos')
    parse.add_argument('--test', action='store_false')
    args = parse.parse_args()
    return args

args = parse_args()
print(args.test)