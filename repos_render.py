## Author: Weishi Ding, Zichen Yang
## Date: 4/10/2024
## Description: 
## This file will dynamically generate the HTML code that displays
## the all user's github repos on the frontend


def generate_repos_html(repos_json):
    html = ""
    
    for repo_name in repos_json:
        html += f'<div class="folder collapsed" id="{repo_name}" onclick="toggleRepo(this, event)"> \n'
        # Each repo is a toggleable rectangle
        html += f'<div class="folder-name"><span>{repo_name}</span><span class="folder-toggle">&#9658;</span></div> \n'
        html += f'  <div class="files-container" id="inner-{repo_name}" style="display: none;"> \n'
        # Content to be toggled can go here
        html += f'  </div>\n'
        html += '</div>\n'

    return html


def generate_all_html(repos):
    html = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, height=device-height, initial-scale=1.0" crossorigin="anonymous"> 
        <script src="https://cdn.jsdelivr.net/npm/marked@3.0.7/marked.min.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <!-- Include the Highlight.js library -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.1/highlight.min.js"></script>
        <!-- Include a Highlight.js theme for styling -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.1/styles/default.min.css">
            <!-- Initialize Highlight.js -->
        <script>hljs.highlightAll();</script>

        <script>
            window.onload = function() {
            localStorage.clear();
            sessionStorage.clear();
            };
        </script>

        <title>Code Editor</title>
        <style>
            body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: #333;
            color: white;
            font-size: 18px;
            }

            #container {
            display: flex;
            }

            @media (max-width: 768px) {
            #container {
                flex-direction: column;
                
            }

            .column {
                border-right: none;
            }
            }

            @media (min-width: 769px) {
            #container {
                
            }
            
            .column1 {
                flex-grow: 1;
                max-width: 20%;
                box-sizing: border-box;
                padding: 20px;
                border-right: 1vw solid white;
                display: flex;
                flex-direction: column;
            }

            .column {
                flex-grow: 1;
                max-width: 40%;
                box-sizing: border-box;
                padding: 20px;
                border-right: 1vw solid white;
                display: flex;
                flex-direction: column;
            }
            }

            #sidebar, #code-editor {
            border-right: 2px solid white;
            }

            .github {
            background: none;
            color: white;
            padding: 10px;
            border: 1px solid white;
            cursor: pointer;
            border-radius: 5px;
            margin-bottom: 10px;
            display: block;
            width: 100%;
            text-align: left;
            font-size: 18px;
            }
            
            .github:hover {
            background-color: #55a078;
            }

            .folder {
                margin-bottom: 5px;
                cursor: pointer;
            }

            .folder-name {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 5px;
                border: 1px solid white;
                border-radius: 5px;
                margin-bottom: 5px;
                max-width: 100%;
            }

            .folder-toggle {
                /* Keep your existing styles if any, and ensure it doesn't grow with flex-grow */
                flex-shrink: 0;
            }

            .folder-name:hover {
            background-color: #55a078;
            }
            
            .folder .files-container {
                margin-left: 20px; /* Indent files to visually nest them under the folder */
            }

            .folder.collapsed .file {
            display: none;
            }

            .file {
            margin-left: 20px;
            margin-bottom: 5px;
            cursor: pointer;
            }

            .folder-name, .file {
                flex-grow: 1;
                overflow-wrap: break-word; /* Allows unbreakable words to be broken */
                word-break: break-all;
            }

            .file:hover {
            color: #55a078;
            }

            .tabs {
            display: flex;
            margin-bottom: 10px;
            }

            .tab {
            background: none;
            color: white;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            border: 1px solid white;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
            transition: background-color 0.3s ease, color 0.3s ease;
            }

            .tab:hover {
            background-color: #55a078;
            }

            .tab.active {
            background-color: #66c090;
            }

            .tab-content {
            display: none;
            border: 1px solid white;
            border-radius: 0 0 5px 5px;
            padding: 10px;
            }

            .tab-content.active {
            display: block;
            }
            
            .code-content {
            color: white;
            font-size: 18px;
            line-height: 1.6;
            white-space: pre-wrap;
            }

            .spinner {
            border: 4px solid rgba(0,0,0,.1);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border-left-color: #09f;
            animation: spin 1s ease infinite;
            }

            @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
            }

            #generateButton {
                margin-top: 10px;
                padding: 10px 20px;
                border: none;
                color: white;
                cursor: pointer;
                border-radius:8px;
                
            }

            #downloadButton  {
                margin-top: 10px;
                padding: 10px 20px;
                border: none;
                color: white;
                cursor: pointer;
                border-radius:8px;            
            }

            #downloadFileSpecificButton  {
                margin-top: 10px;
                padding: 10px 20px;
                border: none;
                color: white;
                cursor: pointer;
                border-radius:8px;            
            }

            #generateButton:disabled #downloadButton:disabled #downloadFileSpecificButton:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }

            #generateButton:not(:disabled) {
                background-color: #4CAF50; /* A brighter color when enabled */
            }

            /* Style for the download button when it's enabled */
            #downloadButton:enabled {
                background-color: #008CBA; /* Brighter color for enabled state, e.g., blue */
            }

            /* Style for the download button when it's disabled */
            #downloadButton:disabled {
                background-color: #ccc; /* Gray color for disabled state */
                cursor: not-allowed;
            }

            #downloadFileSpecificButton:enabled {
                background-color: #008CBA; /* Brighter color for enabled state, e.g., blue */
            }

            #downloadFileSpecificButton:disabled {
                background-color: #ccc; /* Gray color for disabled state */
                cursor: not-allowed;
            }

            
        </style>
        </head>
        <body>

        <div id="container">
            
        <div id="sidebar" class="column1">
            <!-- <button onclick="getData()">Get Data from Flask API</button> -->
            <div id="result"></div>
        
            <script>
            // Initialize the repoInfo variable outside of getData to make it globally accessible
            var repoInfo = JSON.parse('{{ repo_info }}');
            // var repoInfo = {{ repo_info | tojson | safe }};
            // var repoInfo = {"sampleKey": "sampleValue"};
            function getData() {
                console.log("Fetched Repo Info:", repoInfo);
                // Assuming repoInfo is an object, you can directly access its properties
                // Here's an example of displaying it as a string in the 'result' div
                document.getElementById('result').innerHTML = JSON.stringify(repoInfo, null, 2);
            }
        </script>

    <button onclick="logout()">Logout</button>
    '''

    html += generate_repos_html(repos)
    html += '''
            </div>

    <div id="code-editor" class="column">
        <div class="code-content" id="code-content">
        <p>Connect to GitHub and open a file</p>
        
        </div>
        <!-- <pre ><code class="code-content python" id="code-content">
        <p>Connect to GitHub and open a file</p> 
        
        </code></pre> -->
    </div>

    <div class="column">
        <div class="tabs">
        <div class="tab active" onclick="showTab('documentation-tab', this)">Documentation</div>
        <div class="tab" onclick="showTab('file-specific-tab', this)">File Specific Summary</div>
        </div>

        <div id="documentation-tab" class="tab-content active">
        <center><p>Please choose a Repo to Generate Documentation.</p></center>
        
        </div>

        <div id="file-specific-tab" class="tab-content">
        <center><p>Please choose a file from a Repo.</p></center>
        </div>

       
        <button id="generateButton" onclick="drive()" disabled>Generate</button>
        
        <button id="downloadButton" onclick="downloadREADME()" disabled>Download README</button>
        <button id="downloadFileSpecificButton" onclick="downloadFileSummary()" disabled hidden>Download File Specific Summary</button>

    </div>
    </div>

    <script>
    let LastClickedRepo = null;
    let LastClickedFile = null;
            
    function check_specific_summary_exist(){
        console.log(`Last repo: ${LastClickedRepo}. Last file: ${LastClickedFile}`);

        $.ajax({
            url: '/check_specific_file_exist?file_name=' + encodeURIComponent(LastClickedFile),
            type: 'POST',
            dataType: 'text',
            success: function(response) {
                var response_txt = JSON.parse(response);
                console.log("External Program Result: " + response_txt.message);
                const htmlContent = marked(response_txt.message);
                document.getElementById('file-specific-tab').innerHTML = htmlContent;
                downloadButton.disabled = false;
                //console.log("Success");
            },
            error: function(error) {
                const htmlContent = `<center><p> Specific file summary for file ${LastClickedFile} in ${LastClickedRepo} does not exist yet. `;
                document.getElementById('file-specific-tab').innerHTML = htmlContent;
                downloadButton.disabled = true;
            }
        });
    }

    function updateButtonsFileSpecific() {
        const downloadFileSpecificButton = document.getElementById('downloadFileSpecificButton');
        const statusMessage = document.getElementById('file-specific-tab'); 
        if(LastClickedFile) {
            downloadFileSpecificButton.disabled = false;
            check_specific_summary_exist();

        } else {
            statusMessage.textContent = "Please choose a file from a Repo.";
        }

    }

    function check_readme_exist(){
        $.ajax({
            url: `/check_readme_exist/${LastClickedRepo}`,
            type: 'POST',
            dataType: 'text',
            success: function(response) {
                var response_txt = JSON.parse(response);
                console.log("External Program Result: " + response_txt.message);
                const htmlContent = marked(response_txt.message);
                document.getElementById('documentation-tab').innerHTML = htmlContent;
                downloadButton.disabled = false;
                
            },
            error: function(error) {
                const htmlContent = `<center><p> Generate Documentation for ${LastClickedRepo} </p></center>`;
                document.getElementById('documentation-tab').innerHTML = htmlContent;
                downloadButton.disabled = true;
            }
        });
    }

    function updateButtonsReadme() {
        const generateButton = document.getElementById('generateButton');
        const statusMessage = document.getElementById('documentation-tab'); 

        if (LastClickedRepo) {
            generateButton.disabled = false; // Enable the button
            check_readme_exist();
           
        } else {
            generateButton.disabled = true; // Disable the button
            downloadButton.disabled = true;
            statusMessage.textContent = "Please choose a repo to generate.";
        }
    }

    function downloadREADME() {
        const content = document.getElementById('documentation-tab').innerText;
        const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `${LastClickedRepo}_README.md`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function downloadFileSummary() {
        const content = document.getElementById('file-specific-tab').innerText;
        const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `${LastClickedFile}_Summary.md`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    
    function toggleRepo(folder, event) {
        var repo_name = event.currentTarget.id;
        LastClickedRepo = repo_name;
        console.log(`Button clicked: {LastClickedRepo}`)
        updateButtonsReadme();
        
        
        event.stopPropagation();
        const isCollapsed = folder.classList.toggle('collapsed');
        const filesContainer = folder.querySelector('.files-container');
        const triangleSymbol = folder.querySelector('.folder-toggle');

        if (isCollapsed) {
            triangleSymbol.innerHTML = '&#9658;'; // Right-pointing triangle
            filesContainer.style.display = 'none'; // Hide files
        } else {
            triangleSymbol.innerHTML = '&#9660;'; // Down-pointing triangle
            filesContainer.style.display = 'block'; // Show files
        }
        
        
        console.log(`This is the repo name ${repo_name}`);
        fetch(`/get-file-html/${repo_name}`)
        .then(response => response.text()) // Assuming the response is directly HTML
        .then(html => {
            document.getElementById(`inner-${repo_name}`).innerHTML = html;
            // Optionally, you might want to toggle visibility instead of reloading every time
        })
        .catch(error => console.error('Error loading repo structure:', error));
    }

    function toggleFolder(folder, event) {
        // add a logic for donwloading the code      
        event.stopPropagation();
        const isCollapsed = folder.classList.toggle('collapsed');
        const filesContainer = folder.querySelector('.files-container');
        const triangleSymbol = folder.querySelector('.folder-toggle');

        if (isCollapsed) {
            triangleSymbol.innerHTML = '&#9658;'; // Right-pointing triangle
            filesContainer.style.display = 'none'; // Hide files
        } else {
            triangleSymbol.innerHTML = '&#9660;'; // Down-pointing triangle
            filesContainer.style.display = 'block'; // Show files
        }
        
    }
    
    async function getFileContent(fileName) {
        try {
        const response = await fetch(fileName);
        
        if (!response.ok) {
            throw new Error('Failed to fetch file');
        }
        
        const fileContent = await response.text();
        return fileContent;
        } catch (error) {
        console.error('Error fetching file:', error.message);
        return null;
        }
    }

    function openFile(fileName, event) {
        console.log(`Opening file: ${fileName}`);
        event.stopPropagation();
        
        getFileContent('/Users/weishiding/Desktop/Capstone/Spring/llmyacc/yacc_code/' + fileName)
        .then(fileContent => {
        // Display the fetched file content in the 'code-content' div
        if (fileContent !== null) {
            document.getElementById('code-content').textContent = fileContent;
        } else {
            document.getElementById('code-content').textContent = "Contents could not be retrieved :(";
        }
        })
        .catch(error => {
        console.error('Error fetching file:', error);
        document.getElementById('code-content').textContent = "Error loading file content.";
        });

        // Render a specific Markdown file in the documentation tab
        // This action does not depend on the code file opened
        renderMarkdownFile('file.md', 'documentation-tab');
    }
    function showTab(tabName, tabElement) {
        const tabs = document.querySelectorAll('.tab-content, .tab');
        tabs.forEach(tab => {
        tab.classList.remove('active');
        });

        // readme 
        const genbutton = document.getElementById("generateButton");
        const readmebutton = document.getElementById("downloadButton");
        const filespecificbutton = document.getElementById("downloadFileSpecificButton");
        
        if (tabName == 'file-specific-tab') {
            genbutton.hidden = true;
            readmebutton.hidden = true;
            filespecificbutton.hidden = false;

        } else {
            genbutton.hidden = false;
            readmebutton.hidden = false;
            filespecificbutton.hidden = true;
        }
        const selectedTab = document.getElementById(tabName);
        selectedTab.classList.add('active');
        tabElement.classList.add('active');

    }

    function drive() {
        var githubUsername = {{ session['github_username'] | tojson }};
        console.log(`Opening file: ${LastClickedRepo}`);
        // spinning effect
        var loadingHtml = '<center><div class="spinner"></div></center>';
        document.getElementById('documentation-tab').innerHTML = loadingHtml;
        $.ajax({
            url: `/run-external-program/${LastClickedRepo}`,
            type: 'POST',
            dataType: 'text',
            success: function(response) {
                var response_txt = JSON.parse(response);
                console.log("External Program Result: " + response_txt.message);
                const htmlContent = marked(response_txt.message);
                document.getElementById('documentation-tab').innerHTML = htmlContent;
                check_readme_exist();
                updateButtonsFileSpecific();

            },
            error: function(error) {
                document.getElementById('documentation-tab').innerHTML = '<center><p>Error loading documentation.</p></center>';
                console.log(error);
            }
        });
        
    }

    function get_file(filename, event) {      
        var file_name = event.currentTarget.id;
        LastClickedFile = file_name;
        updateButtonsFileSpecific();
        event.stopPropagation();

        console.log(`Call get_file(${filename})`)
        $.ajax({
            url: '/get-code?file_name=' + encodeURIComponent(filename),
            type: 'POST',
            dataType: 'json',
            contentType: 'application/json', 
            success: function(response) {
                // Use .html() instead of .text() and apply highlighting
                $('#code-content').html(response.code);
                // Reapply syntax highlighting
                document.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block); // Updated method name
                });   
            },
            error: function(error) {
                console.log(error);
            }
        });
    ;

    }

    function renderMarkdownFile(url, targetElementId) {    
        fetch(url)
        .then(response => {
            if (!response.ok) {
            throw new Error('Failed to fetch Markdown file');
            }
            return response.text();
        })
        .then(markdownText => {
            const htmlContent = marked(markdownText);
            document.getElementById(targetElementId).innerHTML = htmlContent;
        })
        .catch(error => {
            console.error('Error fetching Markdown file:', error.message);
            document.getElementById(targetElementId).innerHTML = "Failed to render documentation";
        });
    }

    function logout() {
        window.location.href = '/logout';
    }
    </script>

    </body>
    </html>
    '''

    return html