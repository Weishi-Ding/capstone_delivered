## Author: Weishi Ding, Zichen Yang
## Date: 4/10/2024
## Description: This file contains the code for generate dependencies
## between code files within a code base. 

import ast
import os
import json
import sys
from git_repo_relation import generate_structure_json
from git_repo_relation import getGitSourceCode


class CallGraphBuilder(ast.NodeVisitor):
    
    def __init__(self):
        self.call_graph = {"imports": []}
        self.current_scope = None

    def visit_Import(self, node):
        '''
        handles 'import module' statements
        These methods add imported module names to the call_graph under imports
        '''
        for alias in node.names:
            self.call_graph["imports"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        '''
        handles 'from ... import ...' statements
        These methods add imported module names to the call_graph under imports
        '''
        self.call_graph["imports"].append(node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        '''
        handles function definitions
        Updates current_scope to the name of the function and adds an entry for 
        this function in the call_graph.
        '''
        func_name = node.name
        self.current_scope = func_name
        self.generic_visit(node)
        self.current_scope = None

    def visit_ClassDef(self, node):
        '''
        Similar to visit_FunctionDef, but for classes.
        Adds an entry for the class, which will include its methods and instances.
        '''
        class_name = node.name
        self.current_scope = class_name
        self.generic_visit(node)
        self.current_scope = None

    def visit_Call(self, node):
        '''
        Processes function/method calls.
        Depending on the current_scope, it updates the call_graph with 
        information about what functions or methods are called within a function 
        or class.
        '''
        if self.current_scope:
            called_function = ast.unparse(node.func)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                # Instance method call
                instance_name = node.func.value.id
                method_name = node.func.attr
                pass
        self.generic_visit(node)

    def visit_Assign(self, node):
        '''
        Identifies instance creations (e.g., obj = MyClass()).
        Records instances of classes being created, which is useful for 
        understanding object usage.
        '''
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            # Instance creation
            class_name = node.value.func.id
        self.generic_visit(node)

def generate_call_graph_git_code(source_code):  
    tree = ast.parse(source_code)
    builder = CallGraphBuilder()
    builder.visit(tree)
    return builder.call_graph

def generate_call_graph(file_path):
    with open(file_path, "r") as source_file:
        source_code = source_file.read()
    tree = ast.parse(source_code)
    builder = CallGraphBuilder()
    builder.visit(tree)
    return builder.call_graph


def construct_paths(json_struct, parent='', current_dir='root'):
    # Initialize the list to store complete file paths
    complete_paths = []
    
    # Add the path prefix for the current directory if it's not the root
    path_prefix = f"{parent}/{current_dir}" if parent else current_dir
    
    # Skip adding the root directory itself to the path
    if current_dir != 'root':
        path_prefix = path_prefix if parent else current_dir
    
    # Get the current directory structure
    current_struct = json_struct[current_dir]
    
    # Add files with the complete path
    for file in current_struct['files']:
        complete_path = f"{path_prefix}/{file}" if parent else file
        complete_paths.append(complete_path)
    
    # Recursively add paths from subdirectories
    for dir in current_struct['dirs']:
        complete_paths.extend(construct_paths(json_struct, path_prefix, dir))
    
    return complete_paths

def collect_local_python_imports(file_paths, import_relation_dict):
    '''
    @file_imports: {"fileName":[number_of_imports, list_of_import_name, file_index]}
    '''
    file_paths = [path for path in file_paths if path.endswith('.py')]
    python_files_without_folder = [path.split('/')[-1].replace('.py', '') for path in file_paths if path.endswith('.py')]
    # print(f"\n the python files without folder are {python_files_without_folder} \n")
    for idx, full_file_path in enumerate(file_paths):
        # with open(file_name + '_git_.json', 'w') as file:
        with open(full_file_path, "r") as file:
            source_code = file.read()
        call_graph = generate_call_graph_git_code(source_code)
        imports = call_graph['imports']
        # print(f"\n the current file imports are {imports} \n")
        clean_imports = list(set(python_files_without_folder).intersection(set(imports)))
        # print(f"\n the current file imports (clened) are {clean_imports} \n")
        import_relation_dict[full_file_path] = [len(clean_imports), clean_imports, idx] 
        
    return import_relation_dict


# construct the file dependency after downloading github files to my local storage
def construct_dependency(BASE_DIR, owner, repo, token):
    structure_file_name = token + "folder_structure.json"
    storage_repo_path = os.path.join(BASE_DIR, owner, repo)
    with open(os.path.join(storage_repo_path, structure_file_name), 'r') as file:
        repo_structure_dict = json.load(file)
    # transform the repo_structure_dict to a list of all paths
    file_paths = construct_paths(repo_structure_dict, parent='', current_dir='root')
    file_paths = [path.replace('root/', '') for path in file_paths]
    file_paths = [f"{storage_repo_path}/{path}" for path in file_paths]

    import_relation_dict = {}
    collect_local_python_imports(file_paths, import_relation_dict)
    with open('dependency_relation_for_visualization.json', 'w') as json_file:
        json.dump(import_relation_dict, json_file, indent=4)
    return (import_relation_dict)


