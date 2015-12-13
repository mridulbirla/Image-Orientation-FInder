__author__ = 'BlackPanther'

import math
import operator
import collections



def dataprocessing(fname, data):
    with open(fname) as f:
        for line in f:
            data.append([n if '.jpg' in n else int(n) for n in line.strip().split(' ')])

def heuristic_euclideandistance(image1, image2):
    euc_distance = 0
    for x in range(2, len(image1)):
        euc_distance += abs(((image1[x]) - (image2[x])))
    return (euc_distance)

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
    for x in range(len(testdata)):
        try:
            if testdata[x][1] == testdata[x][len(testdata[x])-1]:
                right += 1
            else:
                print "Wrong assignment"
        except ValueError:
            print "phat gaya BC"

    return (right/float(len(testdata))) * 100

# def backwardelimination(traindata, reduced_)

def normalize(data):
    max = 0
    min = 1000000
    for each in ((data)):
        for every in range(2, len(each)):
            if each[every] > max:
                max = each[every]
            if each[every] < min:
                min = each[every]

    for each in ((data)):
        for every in range(2, len(each)):
            each[every] = (each[every] - min) / (max - min)

def main():
    traindata = []
    testdata =[]

    traindata_beforepreprocessing = []
    testdata_beforepreprocessing = []

    dataprocessing('train-data.txt', traindata_beforepreprocessing)
    dataprocessing('test-data.txt', testdata_beforepreprocessing)


    for y in range(len(traindata_beforepreprocessing)):
        temp = []
        x = 0
        while x < len(traindata_beforepreprocessing[y]):
            if x == 0:
                temp.append(traindata_beforepreprocessing[y][x])
                temp.append(traindata_beforepreprocessing[y][x+1])
            else:
                temp.append(0.2989*traindata_beforepreprocessing[y][x] + 0.5870*traindata_beforepreprocessing[y][x+1] + 0.1140*traindata_beforepreprocessing[y][x+2])
            if x == 0:
                x += 2
            else:
                x += 3
        traindata.append(temp)
    x = 0

    for y in range(len(testdata_beforepreprocessing)):
        temp = []
        x = 0
        while x < len(testdata_beforepreprocessing[y]):
            if x == 188:
                print "Counting Starts"
            if x == 0:
                temp.append(testdata_beforepreprocessing[y][x])
                temp.append(testdata_beforepreprocessing[y][x+1])
            else:
                temp.append(0.2989*testdata_beforepreprocessing[y][x] + 0.5870*testdata_beforepreprocessing[y][x+1] + 0.1140*testdata_beforepreprocessing[y][x+2])
            if x == 0:
                x += 2
            else:
                x += 3
        testdata.append(temp)

    # normalize(testdata)
    # normalize(traindata)

    k = 101


    for each in range(len(testdata)):
        nearest_neigbors = k_nearest_neighbors(traindata, testdata[each], k)
        label_assigned = get_label_assigned(nearest_neigbors)
        testdata[each].append(label_assigned)
        print('class label assigned' , label_assigned , 'actual label', testdata[each][1])

    accuracy_value = accuracy(testdata)

    print ('Accuracy: ' + str(accuracy_value) + '%')

main()

#   intensity = 0.2989*red + 0.5870*green + 0.1140*blue
#  Y = 0.2126 R + 0.7152 G + 0.0722 B
