import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)

args = parser.parse_args()

for dir_path, dir_names, file_names in os.walk(args.path):
    #print (dir_path, dir_names)
    for f in file_names:
        print (os.path.join(dir_path, f))
