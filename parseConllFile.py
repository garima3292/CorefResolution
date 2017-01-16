import csv
import re
from collections import defaultdict

mentionsList = {}
wordsList = []

mentionsFile = open('mentionsList.txt', 'w')
wordsFile = open('wordsList.txt', 'w')

mentionOpen = ''
activeMentions = []
activeMentionWords = []

def parseCorefExpression(corefExpression):
	tempStr = corefExpression[:]
	tempStr = re.sub('\(|\)','', tempStr)
	return int(tempStr)
#endDef
	
#(2(3), (2, (2) are open mention case; count >=0
#(2)3), 2), 2)3) are closed	mention case; count < 0
def matchMentionsCase(corefExpression):
	count = 0
	for char in corefExpression:
		if char == '(':
			count += 1
		elif char == ')':
			count -= 1
	return count
#endDef	

#this mention has been extracted, modify the global object to reflect it
def handleEndMentionCase(corefExpression):
	mentionCluster = parseCorefExpression(corefExpression)
	indexInList = activeMentions.index(mentionCluster)
	key = ('_').join(activeMentionWords[indexInList])
	mentionsList[key] = mentionCluster
	wordsList.append(key)

	mentionsFile.write('%s %d' %(key, mentionCluster))
	mentionsFile.write('\n')
	wordsFile.write('%s' %(key))
	wordsFile.write('\n')
	print "final coref chain from open case"
	print activeMentionWords[indexInList]
	print mentionCluster
	
	activeMentionWords.pop(indexInList)
	activeMentions.pop(indexInList)
	if (len(activeMentions) == 0):
		mentionOpen = False
#endDef

def parseConn():
	f = open('testConll.conll', 'r')
	conllLines = f.readlines()
	mentionOpen = False

	for connLine in conllLines:
		cols = connLine.split()
		colsize = len(cols)
		if(colsize < 7):
			continue
		else:
			word = cols[3]
			corefExpression = cols[colsize-1]
			
			if (corefExpression == '-'):
				if(mentionOpen == True):
					for activeMention in activeMentionWords:
						activeMention.append(word)
				else:
					if('/.' not in word):
						wordsList.append(word)
						wordsFile.write('%s' %(word))
						wordsFile.write('\n')
			elif (matchMentionsCase(corefExpression) >= 0):
				
				mentionOpen = True
				if('|' in corefExpression):
					corefsArr = corefExpression.split('|')
				else:
					corefsArr = []
					corefsArr.append(corefExpression)

				for coref in corefsArr:	
					mentionCluster = parseCorefExpression(coref)
					activeMentions.append(int(mentionCluster))
					activeMentionWords.append([])

				for activeMention in activeMentionWords:
					activeMention.append(word)

				# print activeMentionWords
			
				for coref in corefsArr:
					if (coref.endswith(')')):
						handleEndMentionCase(coref)

			elif (matchMentionsCase(corefExpression) < 0): #ends with )
				if('|' in corefExpression):
					corefsArr = corefExpression.split('|')
				else:
					corefsArr = []
					corefsArr.append(corefExpression)

				if('(' in corefExpression):
					id = parseCorefExpression(corefsArr[0])
					activeMentions.append(id)
					activeMentionWords.append([])
				
				for activeMention in activeMentionWords:
					activeMention.append(word)

				for coref in corefsArr:
					handleEndMentionCase(coref)

#endDef

parseConn()
