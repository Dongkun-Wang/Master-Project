#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:38:21 2019

@author: chenxi
"""

import jieba.posseg as pseg
import re
import os


def get_jay_files(jay_path='jay_clean/'):
    jay_files =[]
    for fpathe, dirs, fs in os.walk(jay_path):
        for f in fs:
            if 'txt' in f:
                jay_files.append(os.path.join(fpathe,f))
    return jay_files

def get_sentences(jay_files):
    sentences = []
    for jay_file in jay_files:        
        with open(jay_file,'r') as f:
            contents = f.readlines()
    
        for content in contents:
            temp = re.split(' |\n', content)
            for _ in temp:
                if len(_):
                    sentences.append(_)
    return sentences

def get_words_list(sentences):
    '''
    inputs: sentences
    outputs: words_list
    '''
    words_list = []
#    flags_list = []
    for sentence in sentences:
        temp = pseg.cut(sentence)
        for word, _ in temp:
            words_list.append(word)
#            flags_list.append(flag)
    return list(set(words_list)) #, flags_list

def get_jay_words(jay_path):
    sentences = get_sentences(get_jay_files(jay_path))
    words_list = get_words_list(sentences)
    return words_list
