#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:26:58 2019

@author: chenxi
"""

from langconv import *

def Traditional2Simplified(sentence):
  '''
  将sentence中的繁体字转为简体字
  :param sentence: 待转换的句子
  :return: 将句子中繁体字转换为简体字之后的句子
  '''
  sentence = Converter('zh-hans').convert(sentence)
  return sentence

def Simplified2Traditional(sentence):
  '''
  将sentence中的简体字转为繁体字
  :param sentence: 待转换的句子
  :return: 将句子中简体字转换为繁体字之后的句子
  '''
  sentence = Converter('zh-hant').convert(sentence)
  return sentence

if __name__=="__main__":
  traditional_sentence = '憂郁的臺灣烏龜'
  simplified_sentence = Traditional2Simplified(traditional_sentence)
  traditional_sentence = Simplified2Traditional(simplified_sentence)
  print(simplified_sentence)
  print(traditional_sentence)