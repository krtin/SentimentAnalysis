from nltk import word_tokenize
from nltk import pos_tag
import pandas as pd
import re
import string
from nltk.corpus import stopwords
import enchant
import unicodedata

dictionary = enchant.Dict("en_US")
stoplist = set(stopwords.words('english'))
emoticonfile = "../data/emoticon.csv"
emoticons = pd.read_csv(emoticonfile,header=None)
acronymfile = "../data/acronyms.csv"
acronyms = pd.read_csv(acronymfile)
contractionfile = "../data/contractions.csv"
contractions = pd.read_csv(contractionfile,header=None)
dalscorefile = "../data/dalscores.csv"
dalscores = pd.read_csv(dalscorefile)
puncs = string.punctuation
punc_map = [len(puncs)]
punc_map = ["EXC","DQOU","HASH","DOL","PER","AND","QOU","PARO","PARC","MUL","ADD","COM","SUB","DOT","FORS","COL","SEMI","LESS","EQL","GREAT","QUES","ATT","SQRO","BSLASH","SQRC","EXP","UNDER","ACC","CURO","PIPE","CURC","TILD"]



def checkwordf(word):
	wispunc=True
	try:
		i=puncs.index(word)
	except:
		wispunc=False
	if(word in puncs):
		return punc_map[i]
	elif(word in stoplist):
		return "tree(STOP,tree("+word+"))"
	elif(dictionary.check(word) and word!="..."):
		pos=pos_tag([word])
		pos=pos[0][1]
		#f = open('../data/wordlist.csv','a')
		#f.write(word+'\n')
		#f.close()
		polarity="NEU"
		lowercaseword=word.lower()
		if(any(dalscores["word"]==lowercaseword)):
			score = dalscores[dalscores["word"]==lowercaseword]["score"].as_matrix()[0]
			if(score>0.9):
				polarity="POS"
			elif(score>0.8):
				polarity="POS"
			elif(score<0.2):
				polarity="NEG"
			elif(score<0.4):
				polarity="NEG"
			elif(score<0.5):
				polarity="NEG"

		#file = open("../data/wordlist.txt","a")
		#file.write(word+"+")
		return "tree(EW,tree("+pos+","+lowercaseword+","+polarity+"))"
	else:	
		return "tree(NE,tree("+word+"))"



def checkword(word):
	#print(word[0])
	if(any(emoticons[0]==word)):
		value = emoticons[emoticons[0]==word][1].as_matrix()[0]
		if(value==1):
			tag="||EP||"
		elif(value==0.5):
			tag="||P||"
		elif(value==0):
			tag = "||NE||"
		elif(value==-0.5):
			tag = "||N||"
		else:
			tag="||EN||"
		return True, tag
	elif(word[0]=="@" or word[0]=="#"):
		return True, "||T||"
	elif(len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', word))>0):
		return True, "||U||"
	elif(any(contractions[0]==word)):
		word = contractions[contractions[0]==word][1].as_matrix()[0]
		return False, word
	else:
		return False, word

def partialsent(sent):
	
	sent = sent.split(" ")
	outsent = []	
	newsent = []
	for word in sent:
		found,tag = checkword(word)
		if(found):
			outsent.append(tag)
		else:
			newsent.append(tag)
	return outsent,newsent 
				
def parsesent(newsent,outsent):	
	sent = word_tokenize(newsent)
	finalsent=[]
	
	for word in sent:
		if(word==""):
			continue
		if(any(acronyms["acronym"]==word)):
			
			word=word_tokenize(acronyms[acronyms["acronym"]==word]["value"].as_matrix()[0])
			
			if(len(word)==1):
				word = checkwordf(word[0])
				#print(word)
				finalsent.append(word)
			else:
				#print(word)
				#contains multiple words after expansion
				newword=[]
				for w in word:
					w = checkwordf(w)
					newword.append(w)
				finalsent.extend(newword)
		else:
			word = checkwordf(word)
			finalsent.append(word)
	#print(finalsent)	
	return outsent+finalsent

def makepseudotree(sents):
	#.encode('ascii', 'ignore')
	i=1
	total=len(sents)
	for sent in sents:
		sent=sent.decode('utf-8','ignore')
		sent=sent.encode('ascii','ignore')
		sent=sent.strip()
		#print(sent)
		
		outsent,newsent = partialsent(re.sub(' +',' ',sent))
		
		#print(newsent)
		finalsent = parsesent(" ".join(newsent),outsent)
		#print(finalsent)
		#print(i,total)
		i=i+1
		finalsent=",".join(finalsent)
		finalsent = "tree("+finalsent+")"
		print(finalsent)		
			

