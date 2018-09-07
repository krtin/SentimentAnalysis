from bs4 import BeautifulSoup
import urllib2
import pandas as pd
index=["1","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
url="http://www.noslang.com/dictionary/"
data=[]
data.append(["acronym","value"])
for i in index:
	print(i)
	hdr = {'User-Agent':'Mozilla/5.0'}
	req = urllib2.Request(url+i,headers=hdr)

	page = urllib2.urlopen(req)
	html = page.read()
	
	soup = BeautifulSoup(html, 'html.parser')
	#print(soup.prettify())
	
	mydivs = soup.findAll("div", { "class" : "dictonary-word" })
	for div in mydivs:
		data.append([div.contents[0]["name"],div.contents[1]["title"]])
	
data = pd.DataFrame(data)
	
data.to_csv("../data/acronyms.csv",header=False,index=False)
