## Author: Weishi Ding, Zichen Yang
## Date: 4/10/2024
## Description: This is a visualizer to test the dependency tree constructed 
## from our parse; this is only used during testing not in production code;
## but feel free to use it

import json
import os
from graphviz import Digraph

# the path where the dependency is stored, please replace it with the actual dependency JSON file you generated
# the generation of dependency JSON file is in line 159 of the parse.py
path = '/Users/weishiding/Desktop/Capstone/Spring/llmyacc/yacc_code/dependency_relation_for_visualization.json'

with open(path, 'r') as json_file:
    dependencies = json.load(json_file)

def create_dependency_graph(dependencies):
    dot = Digraph(comment='File Dependency Diagram')

    # Mapping for full file names to base file names
    full_to_base = {path: os.path.basename(path).replace('.py', '') for path in dependencies}

    # Add nodes
    for base_name in full_to_base.values():
        dot.node(base_name, base_name)

    # Add edges based on dependencies
    for full_path, data in dependencies.items():
        base_file = full_to_base[full_path]
        for dep in data[1]:
            # Check if the dependency is a file in the dictionary
            if dep in full_to_base.values():
                dot.edge(dep, base_file)

    return dot

# Create the graph
graph = create_dependency_graph(dependencies)

# Render the graph to a file
graph.render('file_dependency_diagram', view=True)  # This will also automatically open the diagram

# If you just want to output the diagram source
print(graph.source)

