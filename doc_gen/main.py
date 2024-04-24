## Author: Weishi Ding, Zichen Yang
## Date: 4/10/2024
## Description: This file is the driver program for generating
## a documentation for a code base. For a code file, LLM will
## generate a file specific documentation, and then use this 
## to generate a running summary, which contains all files'
## information that LLM has reasoned so far. Finally, it will
## generate a holistic documentation for the entire code base.


import json
import os
import heapq
from openai import OpenAI
from parse import construct_dependency
import math


import tiktoken

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

PROMPT_BASE_DIR = '/Users/weishiding/Desktop/Capstone/Spring/llmyacc/yacc_code/doc_gen/prompt'
context_window_dict = {"gpt-4-0613": 4096, "gpt-4-turbo-2024-04-09": 4096, "gpt-3.5-turbo": 4096, "mistral_8x7b": 2048}
gpt4_family = set(["gpt-4-0613", "gpt-4-turbo-2024-04-09"])
gpt35_family = set(["gpt-3.5-turbo"])
client = OpenAI(api_key='replace with your own key')

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    '''
    encoding_name: different for different model
    From: https://stackoverflow.com/questions/75804599/openai-api-how-do-i-count-tokens-before-i-send-an-api-request
    '''
    if(encoding_name == "gpt-4-0613"):
        encoding = tiktoken.encoding_for_model(encoding_name)
    elif(encoding_name == "gpt-4-turbo-2024-04-09"):
        encoding = tiktoken.encoding_for_model("gpt-4")
    elif(encoding_name == "gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    elif(encoding_name == "mistral_8x7b"):
        num_tokens = math.floor(len(string) / 0.75)
    num_tokens = len(encoding.encode(string))
    num_tokens = math.floor(num_tokens * 1.07)
    return num_tokens

def tokens_to_chars(num, multplier):
    return math.ceil(num * multplier)

def chunk_size(model, ratio = 4):
    '''
    return the chunk size (in chars) based on consideration of the model
    '''
    # a rough gauge of the number of tokens consumed
    return tokens_to_chars(context_window_dict[model], ratio)

def chunk_decider(model_name, source_input):
    '''
    calculate the rough number of tokens and lines in a python file
    input@model: model name 
    input@source_input: a string
    output@(decision, chunk_size)
    decision: True or False
    '''
    total_num_token = num_tokens_from_string(source_input, model_name)
    token_limit = context_window_dict[model_name]
    # print(f"Total num of tokens in this file is {total_num_token}, token limit is {token_limit}")
    if(total_num_token > token_limit):
        return True
    else:
        return False

def build_heap_ele(depend_dic):
    heap_ele = list(depend_dic.values())
    heapq.heapify(heap_ele)
    
    return heap_ele



# file specific prompt for first file and all other files
def generate_documentation_prompt(code, count, summary, file_name):
    # print(f"current specific summary we are generating for is {file_name} \n")
    if count == 0: # this is the first file for us to generate
        prompt_path = os.path.join(PROMPT_BASE_DIR, 'initial_prompt_file_specific.txt')
        with open(prompt_path,'r') as file:
            prompt = file.read()
        return prompt.format(code = code)
    else: 
        prompt_path = os.path.join(PROMPT_BASE_DIR, 'ongoing_prompt_file_specific.txt')
        with open(prompt_path,'r') as file:
            prompt = file.read()
        return prompt.format(summary = summary, code = code)


def get_code_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def get_response(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model,
    messages=messages,
    temperature=0.1)
    return response.choices[0].message.content


'''
  code_split will be called if the given file is too large to be fit into the context window.
  The language is Python.
  The default chunk size is 8000 characters.
  8000 char IS NOT 8000 tokens
  The function do the following steps:
  1. Load the file from the given path.
  2. Split the files into general chunks.
  3. Split the chunks into even smaller chunks.
  3. Return the smaller chunks.
'''
def code_split(file_path, parser_threshold_num=0, chunk_size_num=8000):
    loader = GenericLoader.from_filesystem(
        file_path,
        glob="*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=parser_threshold_num),
    )
    docs = loader.load()


    py_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=chunk_size_num, chunk_overlap=0
    )
    result = py_splitter.split_documents(docs)

    return result

def generate_large_file_summary(file_path, model_name):
    count = 0
    summary = ""
    code = ""
    chunks = code_split(file_path, 0, chunk_size(model_name))

    for chunk in chunks:
        code = chunk.page_content
        prompt = get_chunks_prompt(count, code, summary)
        description = get_response(prompt, model_name)
        summary = description
        # print(f"----This is the generation for chunk {count}----")
        # print(description)
        count += 1
    res = summary.split('- - - - - - - - - - - - - - - - - -')
    return res[-1]

def get_chunks_prompt(count, code, summary):
    if count == 0:
        prompt_path = os.path.join(PROMPT_BASE_DIR, 'initial_prompt_chunk.txt')
        with open(prompt_path,'r') as file:
            prompt = file.read()
        return prompt.format(code = code)
    else:
        prompt_path = os.path.join(PROMPT_BASE_DIR, 'ongoing_prompt_chunk.txt')
        with open(prompt_path,'r') as file:
            prompt = file.read()
        return prompt.format(summary = summary, code = code)


def generate_documentation_prompt_sum_chunk(model, description, summary, count, file_name):
    print(f"current (chunk) running summary we are generating for is {file_name} \n")
    if count == 0:
        prompt_path = os.path.join(PROMPT_BASE_DIR, 'initial_prompt_chunk_running_summary.txt')
        with open(prompt_path,'r') as file:
            prompt = file.read()
        return prompt.format(description = description)

    else:
        prompt_path = os.path.join(PROMPT_BASE_DIR, 'ongoing_prompt_chunk_running_summary.txt')
        prompt_path_gpt4 = os.path.join(PROMPT_BASE_DIR, 'ongoing_prompt_chunk_running_summary_GPT4.txt')
        if (model in gpt4_family):
            with open(prompt_path_gpt4,'r') as file:
                prompt = file.read()
        elif(model in gpt35_family):
            with open(prompt_path,'r') as file:
                prompt = file.read()
        return prompt.format(summary = summary, description = description)
    
# running summary prompt for first file and all other files
def generate_documentation_prompt_sum(model, description, count, summary, file_name):
    print(f"current running summary we are generating for is {file_name} \n")
    
    if count == 0: # running summary for the first file
        prompt_path = os.path.join(PROMPT_BASE_DIR, 'initial_prompt_running_summary.txt')
        with open(prompt_path,'r') as file:
            prompt = file.read()
        return prompt.format(description = description)
    
    else: # running summary for ongoing files
        prompt_path = os.path.join(PROMPT_BASE_DIR, 'ongoing_prompt_running_summary.txt')
        prompt_path_gpt4 = os.path.join(PROMPT_BASE_DIR, 'ongoing_prompt_running_summary_GPT4.txt')
        if (model in gpt4_family):
            with open(prompt_path_gpt4,'r') as file:
                prompt = file.read()
        elif(model in gpt35_family):
            with open(prompt_path,'r') as file:
                prompt = file.read()
        return prompt.format(summary = summary, description = description)
        

def generate_finaldoc_prompt(summary):
    prompt_path = os.path.join(PROMPT_BASE_DIR, 'final_prompt.txt')
    with open(prompt_path,'r') as file:
        prompt = file.read()
    return prompt.format(summary = summary)


def mainDriver(model_name_each_file, model_name_running_summary, BASE_DIR, owner, repo, token):
    # @model_name_each_file: This is the name of the model that will be used for each file's specific summary
    # @model_name_final: This is the name of the model that will be used for running summary
    # depend_dic is a dictionary that stores the dependcy of each files
    # The value storeed in this dictionary is the following:
    # {file_name: [#imports, [imports names], index in python_files]}
    # #imports will decreased by one if one of its dependency file's documentation
    # has been generated
    # heap_ele is a heap whoes elementes are the values in the depend_dic
    # It's heapified according to the #imports of each value
    heap_ele = []

    depend_dic = construct_dependency(BASE_DIR, owner, repo, token)
    heap_ele = build_heap_ele(depend_dic)
    
    count = 0 # number of code files processed so far
    summary = None # running summary so far
    description = None # current file-specific summary
    while heap_ele:
        chosen = heap_ele[0]
        file_name = list(depend_dic.keys())[chosen[-1]]

        with open(file_name, "r") as file:
            code = file.read()
        # current file-specific summary prompt
        prompt = generate_documentation_prompt(code, count, summary, file_name)
        if chunk_decider(model_name_each_file, prompt):
            description = generate_large_file_summary(file_name, model_name_each_file) # file specific summary 
            summary = get_response(generate_documentation_prompt_sum_chunk(model_name_running_summary, description, summary, count, file_name), model_name_running_summary) # running summary
        else:               
            description = get_response(prompt, model_name_each_file) # file specific summary
            # current running summary prompt
            prompt_running_sum = generate_documentation_prompt_sum(model_name_running_summary, description, count, summary, file_name)
            summary = get_response(prompt_running_sum, model_name_running_summary) # running summary
        count += 1

        for idx, ele in enumerate(heap_ele):
            if file_name in ele[-2]:
                heap_ele[idx][0] -= 1
        heapq.heappop(heap_ele)
        heapq.heapify(heap_ele)

        file_specific_name = os.path.join(BASE_DIR, owner, repo, file_name + '_specific_file_summary.txt')
        with open(file_specific_name, 'w') as file:
            file.write(description)
        running_summary_file = os.path.join(BASE_DIR, owner, repo, file_name + '_running_summary.txt')
        with open(running_summary_file, 'w') as file:
            file.write(summary)
    
    print("-----------------------------------This is the final README.md-------------------------------------")
    prompt = generate_finaldoc_prompt(summary)
    readme = get_response(prompt, model_name_running_summary)
    # print(readme)
    return readme



    

