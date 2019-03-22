import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--log-file', type=str, required=True)

args = parser.parse_args()

log_file = os.path.abspath(args.log_file)

fs = open(log_file, 'r')
lines = fs.readlines()
fs.close()

loss_values = []
for j in range(len(lines)):
    line = lines[j]
    if ("Loss" in line):
        line = line.rstrip()
        line = line.split(" : ")
        loss_values.append(line[1].strip())


for j in range(len(loss_values)):
    print (loss_values[j])
