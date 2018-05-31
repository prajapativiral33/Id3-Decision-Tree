import csv
import math
import random
from collections import Counter

attributes = ['clump thickness', 'cell size', 'cell shape','marginal adhesion','single epithelial cell size','bare nuclei','bland chromatin','normal nucleoli','mitoses','class']
#attributes = ['hospitalize','sex','age','case','heal']
#attributes = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']


#attributes = ['sepal length','sepal width','petal length','petal width','class']
#attributes = [ 'buying','maintainance','doors','persons','lug_boot','safety', 'classifications']

#att_values = [[ 'vhigh', 'high', 'med', 'low'],[ 'vhigh', 'high', 'med', 'low'], ['2', '3', '4', '5more'],['2', '4', 'more'], ['small', 'med', 'big'], ['low', 'med', 'high'],['unacc','acc','vgood','good']]

# Class used for learning and building the Decision Tree
#using the given Training Set
class DecisionTree():
    tree = {}
    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)

# Class Node which will be used while classify a
#test-instance using the tree which was built earlier
class Node():
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()

# Majority Function which tells which class has
#more entries in given data-set
def majorClass(attributes, data, target):

    freq = {}
    index = attributes.index(target)

    for tuple in data:
        if (tuple[index] in freq):
            freq[tuple[index]] += 1 
        else:
            freq[tuple[index]] = 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key]>max:
            max = freq[key]
            major = key

    return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetAttr):

    freq = {}
    dataEntropy = 0.0

    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        i = i + 1
    i = i - 1

    for entry in data:
        if (entry[i] in freq):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0

    for freq in freq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return dataEntropy


# Calculates the information gain (reduction in entropy)
#in the data when a particular attribute is chosen for splitting the data.
def info_gain(attributes, data, attr, targetAttr):

    freq = {}
    subsetEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if (entry[i] in freq):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0

    for val in freq.keys():
        valProb        = freq[val] / sum(freq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return (entropy(attributes, data, targetAttr) - subsetEntropy)


# This function chooses the attribute among the remaining
#attributes which has the maximum information gain.
def attr_choose(data, attributes, target):

    best = attributes[0]
    maxGain = 0;

    for attr in attributes:
        newGain = info_gain(attributes, data, attr, target) 
        if newGain>maxGain:
            maxGain = newGain
            best = attr

    return best


# This function will get unique values for that particular
#attribute from the given data
def get_values(data, attributes, attr):

    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values

# This function will get all the rows of the data
#where the chosen "best" attribute has a value "val"
def get_data(data, attributes, best, val):

    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])    
    return new_data


def accuracy_calculation(tree,training_set,test_set,target):
    results = []
    for entry in test_set:
        tempDict = tree.tree.copy()
        result = ""
        while(isinstance(tempDict, dict)):
            root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
            tempDict = tempDict[list(tempDict.keys())[0]]
            index = attributes.index(root.value)
            value = entry[index]
            if(value in tempDict.keys()):
                child = Node(value, tempDict[value])
                result = tempDict[value]
                tempDict = tempDict[value]
            else:
                result = "Null"
                break
        results.append(result == entry[-1])
    accuracy = float(results.count(True))/float(len(results))*100
    return accuracy

#id3_algorithm will generate tree and test it on test_dataset and calculate accuracy based on it

def id3_algorithm(training_set,test_set,target):
    
    tree = DecisionTree()
    tree.learn(training_set, attributes, target )
    
    return accuracy_calculation(tree,training_set,test_set,target)
    
    
#This function will build random forest with 10 trees in it
#Accuracy of each tree and average of all is the output of this tree
def randomForest(training_set, test_set,target):
    K = 10
    trees =[]
    for k in range(K):
        random.shuffle(training_set)
        train=[]
        for i in range(int(len(training_set)/6)):
            train.append(training_set[i])
        tree = DecisionTree()
        tree.learn(train, attributes, target)
        trees.append(tree)
    acc = []
    results = []
    for entry in test_set:
        att_acc=[]
        #total_result =0
        for tree in trees:
            tempDict = tree.tree.copy()
            result = ""
            while(isinstance(tempDict, dict)):
                root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
                tempDict = tempDict[list(tempDict.keys())[0]]
                index = attributes.index(root.value)
                value = entry[index]
                if(value in tempDict.keys()):
                    child = Node(value, tempDict[value])
                    result = tempDict[value]
                    tempDict = tempDict[value]
                else:
                    result = "Null"
                    break
            
            att_acc.append(result == entry[-1])
        #print(att_acc)
        count = Counter(att_acc)
        count_most = count.most_common(1)[0][0]
        results.append(count_most)
        #print(results)
        
        #acc.append(float(att_acc/K)*100)
    #print(len(results))
    #print(results.count(True))
    #print(len(test_set))
    avg_acc = float(results.count(True))/len(test_set)*100
    print ("Average accuracy of random forest is : %.4f" % avg_acc)
    
def build_tree(data, attributes, target):

    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)
        tree = {best:{}}
    
        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree
    print(tree)
    return tree
        

if __name__== "__main__":

    data = []

#Loading CSV file totake input and store in List(Data)
    #with open('car_evaluation.csv') as csvfile:
    with open('Sample.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        
        for row in readCSV:
            #if(row[6] == 'unacc' or row[6] == 'acc'):
            d = row
            data.append(d)
    print('Total instances:',len(data))
    
#Data gets shuffled each time when this code execute
    
    #random.shuffle(data)

#Divide the data into Training Set(80%) and Test Set(20%)
    training_set=[]
    for i in range(0,int(len(data)*0.7)):
        training_set.append(data[i])
    test_set = []
    for i in range(int(len(data)*0.7),len(data)):
        test_set.append(data[i])
    print('Training-set data length :',len(training_set))
    print('Test-set data length :',len(test_set))

    target = attributes[-1]   #Target attribute --last attribute which classify the data

#ID3 algorithm which calculate accuracy
    
    accuracy = id3_algorithm(training_set,test_set,target)
    print('Accuracy of test set is: %.4f'%accuracy)
    accuracy1 = id3_algorithm(training_set,training_set,target)
    print('Accuracy of training set is: %.4f'%accuracy1)
#Random forest function which generate 10 trees
    print('--------------Random Forest----------------')
    randomForest(training_set,test_set,target)


    
    
    
    
   
