 
import math
###----train set------

with open("data/train") as f:
    trainDataSet = f.read()
tarinMailList =  trainDataSet.split('\n')
if(len(tarinMailList[-1])==0):
	tarinMailList.pop()

#global text list variable
totalTrainMail = len(tarinMailList)
uniqueWordTrainSet = set()
trainMailCount = {}
trainMailWordCount = {}

#global probability variable

trainPriorProbability = {}
trainConditionalProbability = {}


for mail in tarinMailList:
	mailType = mail.split(' ')[1]
	trainMailCount.setdefault(mailType,0)
	trainMailCount[mailType]+=1
	trainMailWordCount.setdefault(mailType,{})

	mailTxtList = mail.split(' ')[2:]
	for idx,word in enumerate(mailTxtList):
		if idx%2 == 0:
			uniqueWordTrainSet.add(word)
			trainMailWordCount[mailType].setdefault(word,0)
			trainMailWordCount[mailType][word]+=int(1)


uniqueWordTrainSet = list(uniqueWordTrainSet)
uniqueWordTrainSet.sort()

for word in uniqueWordTrainSet:
	trainMailWordCount["ham"].setdefault(word,0)
	trainMailWordCount["spam"].setdefault(word,0)

#train prior probability
for idx,type in enumerate(trainMailCount):
	trainPriorProbability.setdefault(type,0)
	trainPriorProbability[type] = float(trainMailCount[type]/totalTrainMail)


for type,typeMailTxtList in trainMailWordCount.items():
	trainConditionalProbability.setdefault(type,{})
	for word,count in typeMailTxtList.items():
		trainConditionalProbability[type].setdefault(word,0)
		trainConditionalProbability[type][word] = float((count+1)/float(trainMailCount[type]+len(uniqueWordTrainSet)))


###---start test set----

with open("data/test") as f:
    testDataSet = f.read()
testMailList =  testDataSet.split('\n')
if(len(testMailList[-1])==0):
	testMailList.pop()

currentMailType = []
currentMailName = []
actualMailType = []

for line in testMailList:
	mailType = line.split(' ')[1]
	mailName = line.split(' ')[0]
	currentMailType.append(mailType)
	currentMailName.append(mailName)
	mailTxtList = line.split(' ')[2:]
	

	#apply naive Bayes algorithm
	#P(spam|word1,word2...wordN)
	#use log for avoiding overflow
	probabilityHam = math.log10(trainPriorProbability["ham"])
	probabilitySpam = math.log10(trainPriorProbability["spam"])

	for idx,word in enumerate(mailTxtList):
		if idx%2 == 0:
			if word not in uniqueWordTrainSet :
				probabilityHam += math.log10(1/float(trainMailCount["ham"]+len(uniqueWordTrainSet)))
				probabilitySpam += math.log10(1/float(trainMailCount["spam"]+len(uniqueWordTrainSet)))
			else :
				probabilityHam += math.log10(trainConditionalProbability["ham"][word])
				probabilitySpam += math.log10(trainConditionalProbability["spam"][word])

	if probabilityHam > probabilitySpam:
		actualMailType.append("ham")
	else :
		actualMailType.append("spam")

output = open("data/result",'w')
for i in range(0,len(actualMailType)):
	output.write(currentMailName[i] + " " + currentMailType[i] + " -> " + actualMailType[i] + "\n")

#Analysis Our Total Result

correctSpam = 0
inCorrectSpam =  0
correctHam = 0
inCorrectHam = 0


for i in range(0,len(currentMailType)):
	curr = currentMailType[i]
	predicted = actualMailType[i]
	if(curr == "ham"):
		if(predicted == "ham"):
			correctHam +=1
		else:
			inCorrectHam +=1
	else:
		if(predicted == "spam"):
			correctSpam +=1
		else:
			inCorrectSpam +=1

precision = float(correctSpam/float(correctSpam+inCorrectSpam)) * 100
recall = float(correctSpam/float(correctSpam+inCorrectHam)) * 100
fmeasure = (2*precision*recall)/(precision+recall)
accuracy = ((correctSpam + correctHam)/len(testMailList))*100

print("Ham Email Probability: "+ str(trainPriorProbability["ham"]) +"\nSpam Email Probability: : "+ str(trainPriorProbability["spam"])+'\n')
print("Precision: "+ str(precision) + "\nRecall: "+ str(recall) +"\nF-Measure: "+str(fmeasure)+"\nAccuracy: "+str(accuracy))