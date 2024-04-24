## Author: Weishi Ding, Zichen Yang
## Date: 4/10/2024
## Description: 
## This file will dynamically generate the HTML code that displays
## the all the files within a github repo

def generate_file_html(repo_structure, cur_structure, cur_path):
    html = ""

    # Process directories first
    if "dirs" in cur_structure:
        for dir_name in cur_structure["dirs"]:
            # Recursively generate HTML for subdirectories
            html += f'<div class="folder collapsed" onclick="toggleFolder(this, event)"> \n'
            html += f'  <div class="folder-name"><span>{dir_name}</span><span class="folder-toggle">&#9658;</span></div> \n'
            html += '  <div class="files-container" style="display: none;"> \n'
            html += generate_file_html(repo_structure,repo_structure[dir_name], cur_path + '/' + dir_name)
            html += '  </div> \n'
            html += '</div> \n'
    
    # Then, process files
    if "files" in cur_structure:
        for file_name in cur_structure["files"]:
            html += f'<div id="{cur_path}/{file_name}" onclick="get_file(\'{cur_path}/{file_name}\', event)" class="file">{file_name}</div> \n'

    return html

# Sample repo structure in JSON
repo_structure = {
    "data_processing": {
      "dirs": [],
      "files": [
        "backtest.py",
        "data_process.py"
      ]
    },
    "root": {
      "dirs": [
        "sample_repo"
      ],
      "files": [
        "README.md",
        "Zichen.md"
      ]
    },
    "sample_repo": {
      "dirs": [
        "data_processing"
      ],
      "files": [
        ".DS_Store",
        "DQN.py",
        "agent.py",
        "env.py",
        "main.py",
        "portfolio.py",
        "utils.py"
      ]
    }
}

# Starting point: Generate HTML for the root
html_output = generate_file_html(repo_structure, repo_structure["root"], 'AST_LLM')

