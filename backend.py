## Author: Weishi Ding, Zichen Yang
## Date: 4/10/2024
## Description:
## This file contains the code for the backend part of the project.
## The architecture used is Flask. The OAuth is implemented using Flask_Dance. 

try:
    from flask import Flask,render_template,url_for,request,redirect, make_response, session, flash, jsonify, Markup
    from time import time
    from random import random
    import random
    import requests
    import base64
    import json
    import sys
    sys.path.append('/Users/weishiding/Desktop/Capstone/Spring/llmyacc/yacc_code/doc_gen')
    from main import *
    import nbformat
    from nbconvert import PythonExporter

    from flask import Flask, render_template, make_response
    from flask_dance.contrib.github import make_github_blueprint, github   
    from repos_render import generate_all_html
    from files_render import generate_file_html
except Exception as e:
    print("Some Modules are Missings {}".format(e))


BASE_DIR = '/Users/weishiding/Desktop/Capstone/Spring/llmyacc/yacc_code/storage'
app = Flask(__name__, template_folder='/Users/weishiding/Desktop/Capstone/Spring/llmyacc/yacc_code')
app.config["SECRET_KEY"]="SECRET KEY  "

github_blueprint = make_github_blueprint(client_id='replace with your github client id',
                                         client_secret='replace with your github client secret',
                                         scope = 'repo')

app.register_blueprint(github_blueprint, url_prefix='/github_login')


@app.route('/')
def github_login():
    if not github.authorized:
        return redirect(url_for('github.login'))
    else:
        account_info = github.get('/user')
        if account_info.ok:
            account_info_json = account_info.json()
            session['github_username'] = account_info_json.get('login', 'default_username')
            serialized_json = json.dumps(account_info_json)
            safe_json = Markup(serialized_json)
            repos = user_repos()
            with open('/Users/weishiding/Desktop/Capstone/Spring/llmyacc/yacc_code/homepage.html', 'w') as json_file:
                json_file.write(generate_all_html(repos))
            return render_template('homepage.html', repo_info=safe_json)

        return '<h1>Request failed!</h1>'



@app.route('/token')
def get_token():
    token = session.get('github_oauth_token')
    if token:
        return token
    return 'Token not found!'



@app.route('/user-repos')
def user_repos():
    if not github.authorized:
        return redirect(url_for('github.login'))

    # Fetch the user's repositories
    resp = github.get('/user/repos?type=all')

    if resp.ok:
        repos = resp.json()  # This will be a list of repositories
        # Process the repos as needed, for example, list their names
        repo_names = [repo['name'] for repo in repos]
        return repo_names
    else:
        return jsonify({"error": "Failed to fetch repositories"}), 400
    


#### write folder structure and source code to hardisk
@app.route('/repo-structure/<owner>/<repo>')
def download_repo(owner, repo):
    if not github.authorized:
        return redirect(url_for('github.login'))
    token_info = session.get('github_oauth_token')
    token = token_info['access_token']
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/'
    folder_structure, source_code_dict, all_source_code_dict = generate_structure_json(url, token)
    print(all_source_code_dict.keys())
        
    # save the code dictionary to the designated folder
    for github_path, source_code in all_source_code_dict.items():
        # The github_path would likely need some manipulation if it includes the full GitHub URL; below assumes it's a relative path
        save_file_to_system(owner, repo, github_path, source_code)
    
    # save the JSON file to the same folder
    folder_structure_json = json.dumps(folder_structure, indent=4)
    structure_file_name = str(session.get('github_oauth_token')['access_token']) + "folder_structure.json"
    save_file_to_system(owner, repo, structure_file_name, folder_structure_json)

    # For simplicity, here we just return a simple message
    return jsonify({"message": "Files have been saved successfully."})



def convert_notebook_to_py(notebook_path, output_path):
    # Load the notebook file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = f.read()

    # Convert the string content to a notebook object
    nb = nbformat.reads(notebook_content, as_version=4)

    # Convert to Python script
    py_exporter = PythonExporter()
    python_script, _ = py_exporter.from_notebook_node(nb)

    # Write the Python script to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(python_script)



def save_file_to_system(owner, repo, path, content):
    # Construct the full path where the file will be saved
    # This creates a directory structure like /path/to/your/storage/owner/repo/path/to/file
    full_path = os.path.join(BASE_DIR, owner, repo, path)
    directory = os.path.dirname(full_path)
    os.makedirs(directory, exist_ok=True)

    # Ensure the directory exists
    # Detect if we are writing a JSON file for the folder structure
    if path == (str(session.get('github_oauth_token')['access_token']) + "folder_structure.json"):
        # Write JSON content
        with open(full_path, 'w', encoding='utf-8') as json_file:
            json_file.write(content)
            
    else:
        # Write source code content
        if path.split('.')[-1] == ('ipynb'):
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(content)
            new_path = full_path.split('.')[0] + '.py'
            convert_notebook_to_py(full_path, new_path)
                       
        else:
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(content)
    
    
    


#### This function will be trigged when a repo button is clicked
#### It will download the repo from github and return html code to render the frontend

@app.route('/get-file-html/<repo_name>')
def get_files_html(repo_name):
    repo_path = os.path.join(BASE_DIR, session['github_username'], repo_name)
    structure_file_name = str(session.get('github_oauth_token')['access_token']) + "folder_structure.json"
    try:
        with open(os.path.join(repo_path,  structure_file_name), "r") as file:
            # Load the contents of the file into a Python dictionary
            repo_structure = json.load(file)
            # generate_file_html(structure, cur_path="root", isroot=None)
            if not repo_structure.get('root', None):
                return "<p> This Repo Is Empty </p>"
            return generate_file_html(repo_structure, repo_structure["root"], repo_name)
    except:
        download_repo(session['github_username'], repo_name)
        with open(os.path.join(repo_path, structure_file_name), "r") as file:
            # Load the contents of the file into a Python dictionary
            repo_structure = json.load(file)
            # generate_file_html(structure, cur_path="root", isroot=None)
            if not repo_structure.get('root', None):
                return "<p> This Repo Is Empty </p>"
            return generate_file_html(repo_structure, repo_structure["root"], repo_name)





#### This function is trigger when a file button is clicked
#### It will return the file and display on the front end
@app.route('/get-code',  methods=['POST'])
def get_file():
    filename = request.args.get('file_name', type=str)
    if filename.split('.')[-1] == 'ipynb':
        filename = filename.split('.')[0] + '.py'
    file_path = os.path.join(BASE_DIR, session['github_username'], filename)

    try:
        with open(file_path, 'r') as code_file:
            code_content = code_file.read()
    except:
        # print("Download triggered")
        download_repo(session['github_username'],filename.split('/')[0])
        with open(file_path, 'r') as code_file:
            code_content = code_file.read()
    # Return the content as a JSON response
    return jsonify({"code": code_content})



def revoke_github_token(token):
    requests.post(
        'https://github.com/login/oauth/revoke',
        params={'access_token': token},
        headers={'Authorization': f'token {token}'}
    )


@app.route('/logout')
def logout():
    # print("Session before clearing:", json.dumps(dict(session), indent=4))
    
    # # Explicitly remove the token from Flask-Dance storage
    print(github.token)
    if github.token:
        print("token revocation triggered")
        revoke_github_token(github.token)
        github.token = None

    # session.pop('user_id', None)
    session.pop('github_oauth_token', None)  # Token specific to Flask-Dance
    session.pop('user_id', None)             # Any other user-specific session data
    session.pop('github_username', None)
    session.clear()                          # Clear all remaining session data

    return redirect(url_for('welcome'))


    
@app.route('/welcome')
def welcome():
    print("Session in welcome is :", json.dumps(dict(session), indent=4))
    if github.token:
        print(f"In welcome page the token now is {github.token}")
    return f'''
    <h1>Welcome to Our Application</h1>
    <a href="http://127.0.0.1:5000" target="_blank">Login of GitHub</a>
    '''



@app.route('/check_specific_file_exist',methods=['POST'])
def check_specific_summary_exist():
    filename = request.args.get('file_name', type=str)
    if filename.split('.')[-1] == 'ipynb':
        filename = filename.split('.')[0] + '.py'
    file_path = os.path.join(BASE_DIR, session['github_username'], filename + '_specific_file_summary.txt')

    with open(file_path, 'r') as file:
        result = file.read()

    return jsonify({"message": result})





@app.route('/check_readme_exist/<repo>',methods=['POST'])
def check_readme_exist(repo):
    readme_name = os.path.join(BASE_DIR, session['github_username'], repo, session['github_username'] + 'README' + session['github_username'] + '.md')
    with open(readme_name, 'r') as file:
        result = file.read()
    return jsonify({"message": result})




# Loop through the dictionary
def writeToMemory(documentation_dic, path):
    for file_name, description in documentation_dic.items(): 
        file_path = os.path.join(path, file_name, session['github_username'], '.txt')  
        #Make all the directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        # Open the file for writing
        with open(file_path, 'w') as file:
            # Write the description to the file
            file.write(description)
            
        print(f'File specific summary saved: {file_path}')

@app.route('/run-external-program/<repo>', methods=['POST'])
def run_external_program(repo):
    readme_name = os.path.join(BASE_DIR, session['github_username'], repo, session['github_username'] + 'README' + session['github_username'] + '.md')
    # mainDriver(model_name_each_file, model_name_final, BASE_DIR, owner, repo, token)
    # readme = mainDriver("gpt-4-turbo-2024-04-09", "gpt-4-turbo-2024-04-09", BASE_DIR, session['github_username'], repo, session.get('github_oauth_token')['access_token'])
    # readme = mainDriver("gpt-3.5-turbo", "gpt-3.5-turbo", BASE_DIR, session['github_username'], repo, session.get('github_oauth_token')['access_token'])
    readme = mainDriver("gpt-3.5-turbo", "gpt-4-turbo-2024-04-09", BASE_DIR, session['github_username'], repo, session.get('github_oauth_token')['access_token'])
    with open(readme_name, 'w') as file:
        file.write(readme)
    
    # updated the list of generated repos

    return jsonify({"message": readme})


def parse_file_structure_json(url_init, token, folder_structure, source_code_dict, all_source_code_dict, path="root"):
    if path != "root":
        url_query = f"{url_init}{path}/"
    else:
        url_query = url_init

    # Pass the token to get_files_structure instead of headers
    content = get_files_structure(url_query, token)
    
    if content is None:
        print("Failed to fetch the directory structure.")
        return

    # Iterate through the content
    for item in content:
        current_path = path.split('/')[-1] if path != "root" else "root"
        if current_path not in folder_structure:
            folder_structure[current_path] = {"files": [], "dirs": []}

        if item['type'] == 'file':
            folder_structure[current_path]["files"].append(item['name'])
            cur_url = f"{url_query}{item['name']}"
            # print(cur_url)
            cur_code_file = getGitSourceCode(cur_url, token)

            if isinstance(cur_code_file, bytes):
                # Handle the case where the returned data is bytes
                try:
                    # Try to decode as 'utf-8' if it's expected to be a text file
                    cur_code = cur_code_file.decode('utf-8')
                except UnicodeDecodeError:
                    # If decoding fails, it might be a binary file; encode as base64
                    cur_code = base64.b64encode(cur_code_file).decode('utf-8')
            else:
                # If the returned data is already a string, use it directly
                # print("\n string ever triggered? \n")
                cur_code = cur_code_file


            if cur_url.endswith('.py'):
                source_code_dict[cur_url.replace(url_init, "")] = cur_code
            # if cur_url.split('.')[-1] == ('ipynb'):
            #     print(f"now we are seeing ipynb \n {cur_code} \n ")
            
            all_source_code_dict[cur_url.replace(url_init, "")] = cur_code

        elif item['type'] == 'dir':
            folder_structure[current_path]["dirs"].append(item['name'])
            # Recursively parse the directory structure
            parse_file_structure_json(url_init, token, folder_structure, source_code_dict, all_source_code_dict, item['path'])




def generate_structure_json(url, token):
    # Use the OAuth token from the session
    headers = {'Authorization': f'token {token}'}
    folder_structure = {}  # root is the default parent level folder
    source_code_dict = {}
    all_source_code_dict = {}
    parse_file_structure_json(url, token, folder_structure, source_code_dict, all_source_code_dict, path="root")
    return (folder_structure, source_code_dict, all_source_code_dict)





def getGitSourceCode(api_url, token):
    headers = {'Authorization': f'token {token}'}
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        content_type = api_url.split('.')[-1]  # Extract file extension from URL
        
        content_base64 = response.json()['content']
        if content_base64 == '':
            return 'This file is too big to be fetched from GitHub. You could delete some graphs within this file and try again.'
        
        if content_type == 'ipynb':
            # If the file is a Jupyter Notebook (.ipynb), decode from base64 and then decode to utf-8
            file_content = base64.b64decode(content_base64).decode('utf-8')
            print("\n now we are seeing ipynb \n")
            # print(response.json())
            # print(f"\n now we are seeing ipynb {file_content} \n")
            # strk
            return file_content
        else:
            # For other file types, assume they might need to be treated as binary
            file_content = base64.b64decode(content_base64)
            # print(f"\n now we are seeing py {file_content} \n")
            return file_content
    else:
        print("Failed to fetch file content, status code:", response.status_code)
        return None




def get_files_structure(url, token):
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content
    else:
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    app.run(debug=True)


