import pandas
import random
import numpy
from sklearn import tree
from deepdiff import DeepDiff

# soh enrolacao
dataframe = pandas.read_csv('car.csv', index_col=False, header=0)

data =  dataframe.values

numpy.random.shuffle(data)

leng = len(data)

middle = leng / 2

to_train = data[0:middle]
to_validate = data[middle + 1:leng]

features = map(lambda value: value[0:6], to_train)
labels = map(lambda value: value[6], to_train)

def to_target(value):
	dicionary = {
		'med': 0,
		'vhigh': 1,
		'small': 2,
		'low': 3,
		'unacc': 4,
		'high': 5,
		'big': 6,
		'5more': 7,
		'vgood': 8,
		'good': 9,
		'acc': 10,
		'more': 11
	}

	try:
		return dicionary[value]
	except:
		return value

features_target = map(lambda column: map(lambda value: to_target(value), column), features)

labels_target = map(lambda value: to_target(value), labels)

# treinamento de fato
classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(features_target, labels_target)

to_predict = ["vhigh","med",4,4,"small","high"]

to_predict_target = map(to_target, to_predict)


features_validate = map(lambda value: value[0:6], to_validate)
labels_validate = map(lambda value: value[6], to_validate)

features_validate = map(lambda column: map(lambda value: to_target(value), column), features_validate)
labels_validate = map(lambda value: to_target(value), labels_validate)


result = classifier.predict(features_validate)

cont = 0
for index, value in enumerate(labels_validate):
	if value != result[index]:
		cont += 1

accuracy =  100 - (cont / (len(labels_validate) / 100))

print "A taxa de acuracia eh: " + str(accuracy) + "%"

# print(DeepDiff(features_validate, labels_validate))

