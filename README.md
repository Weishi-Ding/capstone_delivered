# Project Title

AST-Enhanced Automation for Code Base Documentation Using LLM

# Project Description
This project is a proof of concept using language models to holistically document
a large code base. It utilized AST and dependency relation between code files to
construct a dependency tree, then using a modified topological sort algorihm to 
traverse through the dependency tree and drive the language model to reason on 
the code base, suppressing LLM's tendency to hallucinate and bypassing the context
window restriction. 

Please note this project is a proof of concept of framework, and it currently 
ONLY supports .py and .ipynb files as our file dependency construction tool 
currently only support python.


# File and Folder Descriptions
The whole project is under yacc_code folder. It contains the code file for backend 
and frontend implementation.
- `doc_gen` folder
The doc_gen folder inside yacc_code folder implements the code for constructing the
file dependency tree, the code of the modified topologial sort, and the code to drive
the language model to gradually build up the understanding of the code repo.
More specifically, here is a summary of functionality of each file within doc_gen
- `doc_gen/prompt` folder
The prompt folder within the doc_gen folder contains all the prompt we have used; 
you could check where each prompt is used by examining the main.py file.


# Technology Stack
1. Python AST: Used to derive the dependencies between code files within a code base
2. Langchain: Used to slice large code files with syntactic awareness.
3. OpenAI: Used to send request to OpenAI's models
4. tiktoken: Used to calcualte the number of token after embedding text / code to tokens
5. Flask / Flask_dance: Backend architecture and OAuth tools to connect user's github account.
   
# Engineering Highlights
1. Bypassing context window restriction: LLMs tend to hallucinate on the information that
they didn't seen. By letting the LLM reason a code base according to the dependencies between
the code files, we make sure that the LLM will have much information from the depending code files
when it reasons on a specific code file.
1. Scalibity through code slicing: When a code file is too long, it cannot be read by a LLM at once. To
solve this problem, we utilize a syntactic code file slicer that would perform slicing of the code 
file to break it down to smaller chunks and maintaining the code integrity to its best-effort.
This slicer caters to the context window restriction of LLM, together with the dependency tree 
and modified topological sort algorithm, effectively enable the LLM to bypass context window restriction
on code bases with very long code files.
2. Prevention of hallucination: With the modified topological sort algorithm and dependency tree,
we ensure LLM will not hallucinate on content it has yet to seen.
3. Intensive prompt engineering: State of the art prompting strategy, such as chain of thoughts
and one-shot learning, to vastly improve the capability of LLM's on following our instructions,
thus achieving stable and reproducible outcome.
4. Future-proof framework: Although LLMs are rapidly evolving and their context window
lengths are increased in every iteration, the current decoder-only framework suffered
from "needle in a haystack" problem such that the LLM's performance quickly degrades
as input getting longer. In the same time, documenting a code base is a complex task
where even state of the art model such as GPT-4 has trouble following complex instructions 
in one prompt. Our systematic workflow ensures extraction of granular details of codebase
even on rather dated models such as GPT-3.5, and would only perform better as better 
LLMs roll out each year.

# Usage
1. This is a proof of concept project developed locally, thus a lot of the path are
specified to our local development environment. Here are the file with path/configuration
that needs to be modified.
- `main.py`: 
    1. `PROMPT_BASE_DIR` should be replaced by your own path
    2. `OpenAI(api_key=)` should be filled in with your own openai api key
- `backend.py`:
    1. `sys.path.append()` should be replaced by your own path
    2. `BASE_DIR` should be replaced by your own path
    3. `Flask(__name__, template_folder=)` should be replaced by your own path
    4. `github_blueprint(client_id, client_secret)` should be replaced by your own OAuth App Info
    5. On line 51, the path should be replace by your own path. The `homepage.html` is the html file
       dynamically generated according to each user's github repos.
- `call_graph.py`:
    1. `path` should be replaced with your own path with where the actual dependency JSON file you 
        generated, the generation of dependency JSON file is in line 159 of the parse.py

2. To run the web app, please run this command in the terminal before starting the app:
`export OAUTHLIB_INSECURE_TRANSPORT=1`

Then, run: `python backend.py`
The server will be host at 127.0.0.1:5000. 
You will be prompted to login using your github credential. 
Please note this project is a proof of concept of framework, and it currently 
ONLY supports .py and .ipynb files as our file dependency construction tool 
currently only support python.

