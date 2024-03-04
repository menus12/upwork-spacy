#!/usr/bin/env python3

import os
import json
import re
import argparse
import datetime

# Adding command line parameters
parser = argparse.ArgumentParser(description='Upwork data processor')    
parser.add_argument('--file', type=str, help='Projects dump from database')
parser.add_argument('--update', type=bool, help='Update metadata (default is False)')
args = parser.parse_args()

if args.update == None:
    args.update = False

if args.file == None:
    print (parser.print_help())
    exit(1)

start_time = datetime.datetime.now()

f = open(args.file)                 # Opening JSON file
source_file = json.loads(f.read())  # returns JSON object as  a dictionary
f.close()                           # Closing file

print ('File is ' + args.file)
print ('---')



print ('---')
end_time = datetime.datetime.now()
print ('Script has been running for ' + str((end_time - start_time).seconds // 60) + ' minutes ' + str((end_time - start_time).seconds % 60) + ' seconds')
