import pandas as pd
from bs4 import BeautifulSoup
import urllib

url="http://compling.org/cgi-bin/DAL_sentence_xml.cgi?sentence="
filename="../data/wordlist.csv"

#data=pd.read_csv(filename)
f=open(filename,'r')
cols=["word","pleasant"]
data=[]
for word in f:
	word = word.strip('\n').strip('\r')
	if(word==".."):
		continue
	link=url+word
	f = urllib.urlopen(link)           
	myfile = f.read() 
	#print(myfile)
	soup = BeautifulSoup(myfile, 'html.parser')
	if(soup is not None):
		val = soup.word.emotion.measure["valence"]
		if(val!="" and val is not None):
			
			data.append([word,float(val)])
			print(word,val)
	#else:
		#print(word)

data=pd.DataFrame(data)
data.columns = cols
data["pleasant"] = data["pleasant"]/3.0
data.to_csv("../data/dalscores.csv",index=False)

