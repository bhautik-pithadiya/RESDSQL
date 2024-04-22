import os 
import json
import torch
import sqlite3
import argparse
from transformers import (T5TokenizerFast, 
                          T5ForConditionalGeneration)


def parse_option():
    parser = argparse.ArgumentParser('command line arguments for text 2 sql generation.')
    
    parser.add_argument('--device', type = str, default = '0', help =  'the id od used GPU')
    parser.add_argument('--db_path', type=str, default= 'inference_db/test.db',
                        help='database path')
    parser.add_argument('--model_path', type=str, default="models/text2natsql-t5-large/checkpoint-21216",
                        help = "pre-trained model name or path")
    parser.add_argument("--num_beams", type = int, default = 1,
                        help = 'beam size in model.generate() function.')
    parser.add_argument("--num_return_sequences", type = int,default=1,
                        help = 'the number of returned sequences in model.generate() function (num_sequences <= num_beams).')
    
    opt = parser.parse_args()
    return opt 

def get_model(path):
    
    model = T5ForConditionalGeneration.from_pretrained(path)
    if torch.cuda.is_available():
        model = model.cuda()
    
    tokenizer = T5TokenizerFast.from_pretrained(path)
    
    return model, tokenizer

def get_database_info(db):
    conn = sqlite3.Connection(db)
    
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    database_info = ''
    for table in tables:
        table_name = table[0]
        database_info += f"| {table_name} : "
        
        cursor.execute('PRAGMA table_info({})'.format(table_name))
        
        columns = cursor.fetchall()
        
        for col in columns:
            database_info += f"{table_name}.{col[1]}, "
        
        database_info += table_name + ".* "
    
    return database_info

def execute_sql(cursor, sql):
    cursor.execute(sql)    
    
    return cursor.fetchall()

def sql_generation(opt):
    
    model,tokenizer = get_model(opt.path)
    
    database_info = get_database_info(opt.db_path)
    
    question = input("Enter your query - ")
    
    inputs = question + database_info
    
    tokenized_inputs = tokenizer(
        inputs,
        return_tensors='pt',
        padding = 'max_length',
        mex_length = 512,
        truncation = True
    )
    
    encoder_input_ids = tokenized_inputs['input_ids']
    encoder_input_attention_mask = tokenized_inputs['attention_mask']
    
    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
    
    with torch.no_grad():
        model_outputs = model.generate(
            inputs = encoder_input_ids,
            attention_mask = encoder_input_attention_mask,
            max_length = 256,
            decoder_start_token_id = model.config.decoder_start_token_id,
            num_beams = 1,
            num_return_sequences = 1
        )
    
    
    torch.cuda.empty_cache()
    pred_sequence = tokenizer.decode(model_outputs[0], skip_special_tokens = True)
    pred_sql = pred_sequence.split("|")[-1].strip()
    print(pred_sql)
    output = execute_sql(pred_sql)
    print(output)
    
