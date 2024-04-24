## Author: Weishi Ding, Zichen Yang
## Date: 4/10/2024
## Description: This file contains the code for extracting
## the files' relationships within a code base and return a 
## dictionary.


import requests
import json
import base64

def getGitSourceCode(api_url):
    # Send GET request
    response = requests.get(api_url)
    # Check if the request was successful
    if response.status_code == 200:
        # Decode the file content from base64
        file_content = base64.b64decode(response.json()['content'])
        return file_content
    else:
        print("Failed to fetch file content")

def get_files_structure(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content
    else:
        print(f"Error: {response.status_code}")

def parse_file_structure_json(url_init, headers, folder_structure, source_code_dict, all_source_code_dict, path = "root"):
    
    if path != "root": # first time
        url_query = url_init + path + "/"
    else:
        url_query = url_init
    content = get_files_structure(url_query, headers)
    print(f"now the content is {content}")
    for item in content:
        path = path.split('/')[-1]
        if path not in folder_structure:
            folder_structure[path] = {"files": [], "dirs": []}
        if item['type'] == 'file':
            folder_structure[path]["files"].append(item['name'])
            cur_url = url_query + item['name']
            cur_code_response = getGitSourceCode(cur_url)
            try:
                # Try to decode as 'utf-8' if it's expected to be a text file
                cur_code = cur_code_response.decode('utf-8')
            except UnicodeDecodeError:
                # If decoding fails, it might be a binary file; encode as base64
                cur_code = base64.b64encode(cur_code_response).decode('utf-8')

            if cur_url.endswith('.py'):
                source_code_dict[cur_url.replace(url_init, "")] = cur_code
            else:
                all_source_code_dict[cur_url.replace(url_init, "")] = cur_code

        elif item['type'] == 'dir':
            # print(item['name'].split('/'))
            folder_structure[path]["dirs"].append(item['name'])
            parse_file_structure_json(url_init, headers, folder_structure, source_code_dict, all_source_code_dict, item['path'])
            
    return
        
def generate_structure_json(url):
    github_token = "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"
    headers = {'Authorization': f'token {github_token}'}
    folder_structure = {} # root is the default parent level folder
    source_code_dict = {}
    all_source_code_dict = {}
    parse_file_structure_json(url, headers, folder_structure, source_code_dict, all_source_code_dict, path = "root")
    return (folder_structure, source_code_dict, all_source_code_dict)

