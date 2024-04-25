#!/usr/bin/env python3

import yaml
import json
import argparse
from alive_progress import alive_bar

parser = argparse.ArgumentParser(description='NER processor')    
parser.add_argument('--yaml', type=str, help='Filepath for master list')
parser.add_argument('--jsonl', type=str, help='Filepath for resulting list')
args = parser.parse_args()

input = ""
output = ""
master_list = []

if args.yaml is not None:
    input = args.yaml
else:
    exit(1)
    
if args.jsonl is not None:
    output = args.jsonl

def yaml_to_jsonl(yaml_data):
    jsonl_lines = []
    
    # Function to add entries to the JSON Lines list
    def add_entry(label, pattern_words, id_value):
        pattern = [{"LOWER": word.lower()} for word in pattern_words]
        line = json.dumps({"label": label, "pattern": pattern, "id": id_value})
        if line not in jsonl_lines:
            jsonl_lines.append(line)

    with alive_bar(len(yaml_data)) as bar:
        for vendor in yaml_data:
            # Add vendor
            add_entry("VENDOR", [vendor['id']], vendor['id'])
            
            for product in vendor['products']:
                # Add product
                id_value = product['id']
                if vendor['id'] not in id_value and vendor['id'] != 'Misc':
                    id_value = vendor['id'] + ' ' + product['id']
                add_entry("PRODUCT", product['id'].split(), id_value)
                # Handle product aliases as separate entries if they exist
                if 'aliases' in product:
                    for alias in product['aliases']:
                        if alias is not None:
                            add_entry("PRODUCT", alias.split(), id_value)

                # Handle product scopes
                for scope in product.get('scopes', []):
                    add_entry("SCOPE", scope['id'].split(), scope['id'])
                    # Handle scope aliases as separate entries if they exist
                    if 'aliases' in scope:
                        for alias in scope['aliases']:
                            if alias is not None:
                                add_entry("SCOPE", alias.split(), scope['id'])

                for feature in product.get('features', []):
                    # Add feature
                    id_value = feature['id']
                    if product['id'] not in id_value:
                        id_value = product['id'] + ' ' + feature['id']
                    add_entry("FEATURE", feature['id'].split(), id_value)
                    # Handle feature aliases as separate entries if they exist
                    if 'aliases' in feature:
                        for alias in feature['aliases']:
                            if alias is not None:
                                add_entry("FEATURE", alias.split(), id_value)

                    # Handle feature scopes
                    for scope in feature.get('scopes', []):
                        add_entry("SCOPE", scope['id'].split(), scope['id'])
                        # Handle scope aliases as separate entries if they exist
                        if 'aliases' in scope:
                            for alias in scope['aliases']:
                                if alias is not None:
                                    add_entry("SCOPE", alias.split(), scope['id'])
            bar()
    return jsonl_lines

with open(input) as stream:
    try:
        master_list = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

jsonl = yaml_to_jsonl(master_list)

if output != "":
    with open(output, 'w') as file:
        for line in jsonl:
            file.write(line + '\n')