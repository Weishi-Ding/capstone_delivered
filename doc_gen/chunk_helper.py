## Author: Weishi Ding, Zichen Yang
## Date: 4/10/2024
## Description: This file contains the code for slicing the large
## code file. It will calculate the amount of tokens left for the
## context window and decide whether a code file need to be sliced

import tiktoken
import math
from main import generate_documentation_prompt

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    '''
    encoding_name: different for different model
    From: https://stackoverflow.com/questions/75804599/openai-api-how-do-i-count-tokens-before-i-send-an-api-request
    '''
    if(encoding_name == "gpt-4"):
        encoding = tiktoken.encoding_for_model(encoding_name)
    elif(encoding_name == "gpt-4-0613"):
        encoding = tiktoken.encoding_for_model("gpt-4")
    elif(encoding_name == "gpt-4-0125-preview"):
        encoding = tiktoken.encoding_for_model("gpt-4")
    elif(encoding_name == "gpt-3.5-turbo-0125"):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    elif(encoding_name == "mistral_8x7b"):
        num_tokens = math.floor(len(string) / 0.75)
    num_tokens = len(encoding.encode(string))
    num_tokens = math.floor(num_tokens * 1.07)
    return num_tokens

def tokens_to_chars(num, multplier):
    return math.ceil(num * multplier)

def chunk_size(model, num_tokens, overlapping = 0, hardlimit = 8000):
    '''
    return the chunk size (in chars) based on consideration of the model and num_tokens
    @num_tokens: the num token consumed, estimated by tiktoken, should be the whole API call content
    @overlapping: the num tokens that will be overlapping
    '''
    # a rough gauge of the number of tokens consumed
    num_tokens_input = math.floor(num_tokens + overlapping) 
    if (model == "gpt-4"):
        return min(tokens_to_chars((8192 - num_tokens_input), 4), hardlimit)
    elif (model == "gpt-4-0613"):
        return min(tokens_to_chars((8192 - num_tokens_input), 4), hardlimit)
    elif(model == "gpt-4-0125-preview"):
        return min(tokens_to_chars((16384 - num_tokens_input), 4), hardlimit)  
    elif(model == "gpt-3.5-turbo-0125"):
        return min(tokens_to_chars((16384 - num_tokens_input), 4), hardlimit)
    elif(model == "mistral_8x7b"): 
        return min(tokens_to_chars((32768- num_tokens_input), 4), hardlimit)

def chunk_decider(model_name, source_input):
    '''
    calculate the rough number of tokens and lines in a python file
    input@model: model name 
    input@source_input: a list of strings, contain prompt and running summary
    output@(decision, chunk_size)
    decision: True or False
    '''
    dict = {"gpt-4": 8192, "gpt-4-0613": 8192, "gpt-4-0125-preview": 16384, "gpt-3.5-turbo-0125": 16384, "mistral_8x7b": 32768}
    total_num_token = 0
    for each in source_input:
        # num_tokens_from_string(string: str, encoding_name: str) 
        total_num_token += num_tokens_from_string(each, model_name)
    token_limit = dict[model_name]
    if(total_num_token > token_limit):
        return True
    else:
        return False

