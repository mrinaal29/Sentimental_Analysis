#!/usr/bin/env python
# coding: utf-8




#All the important  imports 
import requests
import urllib
from bs4 import BeautifulSoup
import time
import sys
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')
import unicodedata
import re
import pandas as pd
from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import syllables 
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()





final_df=pd.read_csv('dataset.csv') #input data or given data 
df2=pd.read_csv('positive.csv') # positive dictionary 
positives=df2['abound'] 
df1=pd.read_csv('negative.csv',encoding= 'unicode_escape') # negative dictionary 
negatives=df1['2-faced']
url = 'https://www.sec.gov/Archives/' 
df['url']=url+df['SECFNAME'] # combining url to fetch the data 





uncertain=pd.read_csv('uncertainty_dictionary.csv') # uncertainty dictionary of words 
p1 = [p.strip().lower() for p in uncertain['Word']] # strippping the first and the last character ie spaces
constrain=pd.read_csv('constraining_dictionary.csv') # constraining dictionary 
p2 = [j.strip().lower() for j in constrain['Word']]
negative = [neg.strip().lower() for neg in negatives]
positive = [pos.strip().lower() for pos in positives]





# all the functions  to clean the data 
def remove_numbers(text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.\s]' 
    return re.sub(pattern, '', text)


def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)


import nltk
from nltk.tokenize import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
                              # custom: removing words from list
stopword_list.remove('not')
                            # function to remove stopwords
def remove_stopwords(text):
                           # convert sentence into token of words
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
                          # check in lowercase 
    t = [token for token in tokens if token.lower() not in stopword_list]
    text = ' '.join(t)    
    return text

 
def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()



def mang(text, doc_length):
        first_occ = text.find(
            "management's discussion and")+1

        if first_occ != 0:                                      # occurence at index page

            new_text = text[(first_occ)*3:doc_length]                 #skipping the idex
            second_occ = new_text.find(
                "management's discussion and")                                  #title reference

            text_forFlag = text[(first_occ)*3 +
                                second_occ:doc_length]
            flag = text_forFlag.find("item ")                                    

            try:
                mda1 = (text_forFlag[flag+5])                                     # to check if its a title not a word
                mda = (text_forFlag[:flag])

                return mda
            except IndexError:
                return 0
        else:
            return 0




#main function to clean and remove stopwords from the text
def clean_func(text):
    text=remove_special_characters(text)
    text=remove_numbers(text)
    text=remove_stopwords(text)
    text=remove_extra_whitespace_tabs(text)
    return text





# declaring empty list to store values of each score 
pos=[]
neg=[]
pol=[]
a_s_l=[]
per_com=[]
fog_ind=[]
comp=[]
wc=[]
pwp=[]
nwp=[]
us=[]
cs=[]
uwp=[]
cwp=[]
cww=[]

count=0
for i in final_df['url']:
    response = requests.get(i)#scrapping pages into the response tab
    soup = BeautifulSoup(response.text, "html.parser").get_text()#removing html; tags
    txt = str(soup)#converting to string 
    text = txt.lower()
    doc_length=len(text)
    mda = (mang(text, doc_length))
    if mda == 0:
        mda=text   
    tokenized_text_whole = tokenizer.tokenize(text)
    tokenized_text_whole= [wordnet_lemmatizer.lemmatize(w) for w in tokenized_text_whole]
    clean_text=clean_func(mda)# paragraph text
    from nltk.tokenize import RegexpTokenizer   # text to token for finding in dictionaries
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_text = tokenizer.tokenize(clean_text)
    tokenized_text= [wordnet_lemmatizer.lemmatize(w) for w in tokenized_text]

    
    
    def loadPositive(clean_text):# for positive text
        pos = [i for i in clean_text if i in positive]
        return len(pos)
    
    
    def loadNegative(clean_text):# for negative text
        
        neg = [i for i in clean_text if i in negative]
        return len(neg)
    
    
    def complex_words(clean_text):   # syllables count
        count=0
        for i in clean_text:
            if syllables.estimate(i)>2:
                count+=1
        return count
    
    def constrain_words(clean_text): #constrain dic
        con = [i for i in clean_text if i in p2]
        return len(con)
    
    def uncertain_words(clean_text):
        unc = [i for i in clean_text if i in p1]
        return len(unc)
    
    def cons_whole_words(clean_text):
        conw = [i for i in clean_text if i in p2]
        return len(conw)
    
    pos_score=loadPositive(tokenized_text)
    neg_score=loadNegative(tokenized_text)
    complex_count=(complex_words(tokenized_text))
    avg_sen_len=round(len(sent_tokenize(mda))/len(tokenized_text),4)
    per_comp_words=round(complex_count/len(tokenized_text),2)
    fog_index=round(0.4*(avg_sen_len+per_comp_words),2)
    polarity_score=round((((pos_score-neg_score)/(pos_score+neg_score))+0.000001),3)
    word_count=len(tokenized_text)
    pos_word_prop=round((pos_score/word_count),3)
    neg_word_prop=round((neg_score/word_count),3)
    unc_score=uncertain_words(tokenized_text)
    con_score=constrain_words(tokenized_text)
    unc_word_prop=round((unc_score/word_count),3)
    con_word_prop=round((con_score/word_count),3)
    cons_whole_words=cons_whole_words(tokenized_text_whole)
    word_count_cons=len(text)
    constr_whole=round(((cons_whole_words)/word_count_cons),4)
    
    # append all the values into their respective lists 
    pol.append(polarity_score)
    a_s_l.append(avg_sen_len)
    per_com.append(per_comp_words)
    fog_ind.append(fog_index)
    comp.append(complex_count)
    wc.append(word_count)
    pwp.append(pos_word_prop)
    nwp.append(pos_word_prop)
    us.append(unc_score)
    cs.append(con_score)
    uwp.append(unc_word_prop)
    cwp.append(con_word_prop)
    cww.append(cons_whole_words)
    pos.append(pos_score)
    neg.append(neg_score)

    #print( count, pos_score , neg_score , complex_count ,avg_sen_len ,per_comp_words , fog_index,polarity_score,pos_score ,neg_score    ,word_count,pos_word_prop)
    pos_score=0
    neg_score=0





# putting all the list values in a data frame 
df_output = pd.DataFrame({'positive_score':pos,'negative_score':neg,'polarity_score':pol,
                   'average_sentence_length': a_s_l,
                   'percentage_of_complex_words':per_com,'fog_index':fog_ind,'complex_word_count':comp,'word_count':wc,
                   'uncertainty_score': us,
                   'constraining_score':cs,
                    'positive_word_proportion':pwp,
                    'negative_word_proportion':nwp,'uncertainty_word_proportion':uwp,
                   'constraining_word_proportion':cwp,'constraining_words_whole_report':cww})





#concatinating input featurews and output features
final=pd.concat([final_df,df_5],axis=1)
final.head(5)
# getting the output structure
final.to_csv('output.csv')



