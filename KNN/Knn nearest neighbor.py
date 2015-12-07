__author__ = 'BlackPanther'

import math
import operator
import collections



def dataprocessing(fname, data):
    with open(fname) as f:
        for line in f:
            data.append([n for n in line.strip().split(' ')])

def heuristic_euclideandistance(image1, image2):
    euc_distance = 0
    for x in range(2, len(image1)):
        euc_distance += pow((int(image1[x]) - int(image2[x])), 2)
        return math.sqrt(euc_distance)

def k_nearest_neighbors(traindata, testinput, k):
    euc_distances = []
    for x in range(len(traindata)):
        dist = heuristic_euclideandistance(testinput, traindata[x])
        euc_distances.append((traindata[x][1], dist))
    euc_distances.sort(key=operator.itemgetter(1))
    nearest_neighbors = []
    for x in range(k):
        nearest_neighbors.append(euc_distances[x][0])
    return nearest_neighbors

def get_label_assigned(knearestneighbors):
    counter = collections.Counter(knearestneighbors)
    assigned_label = counter.most_common(1)[0][0]
    return assigned_label

def accuracy(testdata):
    right = 0
    for x in range(len(testdata)-1):
        try:
            if testdata[x][1] == testdata[x][194]:
                right += 1
        except ValueError:
            print "phat gaya BC"

    return (right/float(len(testdata))) * 100

def main():
    traindata = []
    testdata =[]

    dataprocessing('train-data.txt', traindata)
    dataprocessing('test-data.txt', testdata)

    k=5


    for each in range(len(testdata)):
        nearest_neigbors = k_nearest_neighbors(traindata, testdata[each], k)
        label_assigned = get_label_assigned(nearest_neigbors)
        testdata[each].append(label_assigned)
        print('class label assigned' + label_assigned + 'actual label' + testdata[each][1])

    accuracy_value = accuracy(testdata)

    print ('Accuracy: ' + repr(accuracy_value) + '%')

main()

#   intensity = 0.2989*red + 0.5870*green + 0.1140*blue
#  Y = 0.2126 R + 0.7152 G + 0.0722 B
