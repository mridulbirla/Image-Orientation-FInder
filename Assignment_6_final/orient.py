import math
import operator
import collections
import sys
import random


train_file_name, test_file_name, algorithm, k_hidden_count_best = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def knn_modified(k_hidden_count_best):
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
            except ValueError:
                print "ERROR"

        return (right/float(len(testdata))) * 100

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

    def update_confusion_matrix(actual_label, assigned_label, confusion_matrix):
        if actual_label == 0:
            if assigned_label ==0:
                confusion_matrix[0][0] += 1
            if assigned_label ==90:
                confusion_matrix[0][1] += 1
            if assigned_label ==180:
                confusion_matrix[0][2] += 1
            if assigned_label ==270:
                confusion_matrix[0][3] += 1
        if actual_label == 90:
            if assigned_label ==0:
                confusion_matrix[1][0] += 1
            if assigned_label ==90:
                confusion_matrix[1][1] += 1
            if assigned_label ==180:
                confusion_matrix[1][2] += 1
            if assigned_label ==270:
                confusion_matrix[1][3] += 1

        if actual_label == 180:
            if assigned_label ==0:
                confusion_matrix[2][0] += 1
            if assigned_label ==90:
                confusion_matrix[2][1] += 1
            if assigned_label ==180:
                confusion_matrix[2][2] += 1
            if assigned_label ==270:
                confusion_matrix[2][3] += 1

        if actual_label == 270:
            if assigned_label ==0:
                confusion_matrix[3][0] += 1
            if assigned_label ==90:
                confusion_matrix[3][1] += 1
            if assigned_label ==180:
                confusion_matrix[3][2] += 1
            if assigned_label ==270:
                confusion_matrix[3][3] += 1

        return confusion_matrix

    def knn_reducedtime(k_hidden_count_best):
        traindata, testdata, traindata_beforepreprocessing, testdata_beforepreprocessing, confusion_matrix = [], [], [], [], [[0,0,0,0], [0,0,0,0],[0,0,0,0], [0,0,0,0]]

        dataprocessing(train_file_name, traindata_beforepreprocessing)
        dataprocessing(test_file_name, testdata_beforepreprocessing)
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
        normalize(testdata)
        normalize(traindata)
        target = open('knn_output_modified.txt', 'w')
        k = k_hidden_count_best
        for each in range(len(testdata)):
            nearest_neigbors = k_nearest_neighbors(traindata, testdata[each], int(k))
            label_assigned = get_label_assigned(nearest_neigbors)
            testdata[each].append(label_assigned)
            target.write(testdata[each][0] + ' ' + str(label_assigned))
            target.write("\n")
            confusion_matrix = update_confusion_matrix(testdata[each][1], label_assigned, confusion_matrix)
        accuracy_value = accuracy(testdata)
        print "Confusion Matrix"
        print ('Accuracy: ' + str(accuracy_value) + '%')
        print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in confusion_matrix]))
        target.close()
    knn_reducedtime(k_hidden_count_best)

def knn(k_hidden_count_best):
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
            except ValueError:
                print "ERROR"

        return (right/float(len(testdata))) * 100

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

    def update_confusion_matrix(actual_label, assigned_label, confusion_matrix):
        if actual_label == 0:
            if assigned_label ==0:
                confusion_matrix[0][0] += 1
            if assigned_label ==90:
                confusion_matrix[0][1] += 1
            if assigned_label ==180:
                confusion_matrix[0][2] += 1
            if assigned_label ==270:
                confusion_matrix[0][3] += 1
        if actual_label == 90:
            if assigned_label ==0:
                confusion_matrix[1][0] += 1
            if assigned_label ==90:
                confusion_matrix[1][1] += 1
            if assigned_label ==180:
                confusion_matrix[1][2] += 1
            if assigned_label ==270:
                confusion_matrix[1][3] += 1

        if actual_label == 180:
            if assigned_label ==0:
                confusion_matrix[2][0] += 1
            if assigned_label ==90:
                confusion_matrix[2][1] += 1
            if assigned_label ==180:
                confusion_matrix[2][2] += 1
            if assigned_label ==270:
                confusion_matrix[2][3] += 1

        if actual_label == 270:
            if assigned_label ==0:
                confusion_matrix[3][0] += 1
            if assigned_label ==90:
                confusion_matrix[3][1] += 1
            if assigned_label ==180:
                confusion_matrix[3][2] += 1
            if assigned_label ==270:
                confusion_matrix[3][3] += 1

        return confusion_matrix


    def knn_basic(k_hidden_count_best):
        traindata, testdata, traindata_beforepreprocessing, testdata_beforepreprocessing, confusion_matrix = [], [], [], [], []

        dataprocessing(train_file_name, traindata)
        dataprocessing(test_file_name, testdata)

        target = open('knn_output.txt', 'w')
        k = k_hidden_count_best
        for each in range(len(testdata)):
            nearest_neigbors = k_nearest_neighbors(traindata, testdata[each], int(k))
            label_assigned = get_label_assigned(nearest_neigbors)
            testdata[each].append(label_assigned)
            target.write(testdata[each][0] + ' ' +str(label_assigned))
            target.write("\n")
            confusion_matrix = update_confusion_matrix(testdata[each][1], label_assigned, confusion_matrix)
        accuracy_value = accuracy(testdata)
        print ('Accuracy: ' + str(accuracy_value) + '%')
        print "Confusion Matrix"
        print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in confusion_matrix]))
        target.close()
    knn_basic(k_hidden_count_best)

LEARNING_RATE=.6

def multiply_matrix(X, Y):
    result = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)] for X_row in X]
    return result


def init_matrix(mat, row, column,val):
    for i in range(row):
        mat.append([val] * column)


def read_input(file_name, input_data,img_names):
    for line in open(file_name, "r"):
        temp = line.rstrip("\n").split(" ")
        if len(temp)==0:
            continue

        p = [0, 0, 0, 0]
        # 0 => [0,0,0,1]   90 => [0,0,1,0] 180 = [0,1,0,0] 270 = [1,0,0,0]
        if int(temp[1]) == 0:
            p[3] = 1
        elif int(temp[1]) == 90:
            p[2] = 1
        elif int(temp[1]) == 180:
            p[1] = 1
        elif int(temp[1]) == 270:
            p[0] = 1
        img_names.append(temp[0])
        t = map(float, temp[2:])
        t.append(1)
        for i in range(0, len(t)):
             t[i] /= 255.00000

        # r_val=t[0::3]
        # g_val=t[1::3]
        # b_val=t[2::3]
        # r_max=max(r_val)
        # r_normalized=map(lambda x: (x/r_max), r_val)
        # g_max=max(g_val)
        # g_normalized=map(lambda x: (x/g_max), g_val)
        # b_max=max(b_val)
        # b_normalized=map(lambda x: (x/b_max), b_val)
        # t=[]
        # for i in range(0,len(r_normalized)):
        #     t.extend([r_normalized[i],g_normalized[i],b_normalized[i]])
        # t.append(1)
        map((lambda x: x/255.0), t)
        q = [t, p]
        input_data.append(q)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def apply_feedforward(input_data, weight_level1, weight_level2, hidden_nodes_count):

    aj = multiply_matrix([input_data], weight_level1)
    # aj2=[]

    # for i in range(hidden_nodes_count+1):
    #     summation=0.00
    #     for j in range(len(input_data)):
    #         summation=summation+(weight_level1[j][i]*input_data[j])
    #     aj2.append(summation)

    for i in range(len(aj[0])):
        aj[0][i] = sigmoid(aj[0][i])
    aj[0][hidden_nodes_count] = 0.001
    aj2 = multiply_matrix(aj, weight_level2)
    for i in range(len(aj2[0])):
        aj2[0][i] = sigmoid(aj2[0][i])
    return aj, aj2


def train_nn(train_data, weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count,img_names):
    one_matrix=[]
    init_matrix(one_matrix,1,output_node_count,1)
    one_matrix2=[]
    init_matrix(one_matrix2,1,hidden_nodes_count+1,1)
    confusion_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    target_file_nn3=open('weight1_output.txt','w')
    target_file_nn2=open('weight2_output.txt','w')
    #target_file_nn=open('nnet_output.txt','w')
    for m in range(0,1):
        correct=0.0
        counter=0
        for example in train_data:
            input_data = example[0]
            expected_output = example[1]
            hidden_layer_output, main_output = apply_feedforward(input_data, weight_level1, weight_level2,
                                                                 hidden_nodes_count)
            #code to
            #this is expected-actual output
            main_index=main_output[0].index(max(main_output[0]))
            actual_index=expected_output.index(1)
            confusion_matrix[actual_index][main_index]+=1
            '''
            if main_index==0:
                target_file_nn.write(str(img_names[counter])+" 270\n")
            elif main_index==1:
                target_file_nn.write(str(img_names[counter])+" 180\n")
            elif main_index==2:
                target_file_nn.write(str(img_names[counter])+" 90\n")
            elif main_index==3:
                target_file_nn.write(str(img_names[counter])+" 0\n")
            '''
            counter+=1
            if main_index==actual_index:
                correct+=1
            delta_main_output=[a_i - b_i for a_i, b_i in zip(expected_output, main_output[0])]
            #this is 1 is actual output
            one_minus_original_output=[a_i - b_i for a_i, b_i in zip(one_matrix[0],main_output[0])]
            mult_term1=[a_i * b_i for a_i, b_i in zip(main_output[0],one_minus_original_output)]
            error_output_node=[a_i * b_i for a_i, b_i in zip(mult_term1,delta_main_output)]
            #print sum(error_output_node)
            one_minus_hidden_output=[a_i - b_i for a_i, b_i in zip(one_matrix2[0],hidden_layer_output[0])]
            mult_term12=[a_i * b_i for a_i, b_i in zip(hidden_layer_output[0],one_minus_hidden_output)]
            error_hidden_node=[]
            for hidden_node_w in range(len(weight_level2)):
                summation=0.0
                for output_node__w in range(len(weight_level2[hidden_node_w])):
                    summation+=weight_level2[hidden_node_w][output_node__w]*error_output_node[output_node__w]
                    weight_level2[hidden_node_w][output_node__w]+= LEARNING_RATE*error_output_node[output_node__w]*hidden_layer_output[0][hidden_node_w]
                summation*=mult_term12[hidden_node_w]
                error_hidden_node.append(summation)

            for input_node in range(len(weight_level1)-1):
                for hidden_node_w in range(len(weight_level1[input_node])):
                    weight_level1[input_node][hidden_node_w]+= LEARNING_RATE*input_data[input_node]*error_hidden_node[hidden_node_w]
            #print str(hidden_layer_output)
        for input_node in range(len(weight_level1)):
            t=""
            for hidden_node_w in range(len(weight_level1[input_node])):
               t+=" "+str(+weight_level1[input_node][hidden_node_w])
            t=t.lstrip(" ")
            t=t+"\n"
            target_file_nn3.write(t)
        target_file_nn3.close()
        for hidden_node_w in range(len(weight_level2)):
            t=""
            for output_node__w in range(len(weight_level2[hidden_node_w])):
                t+=" "+str(+weight_level1[input_node][hidden_node_w])
            t=t.lstrip(" ")
            t=t+"\n"

            target_file_nn2.write(t)
        target_file_nn2.close()
        print "Train Accuracy: " + str(correct/len(train_data) *100)
        print "Confusion Matrix="
        print('\n'.join([''.join(['{:10}'.format(item) for item in row]) for row in confusion_matrix]))
        #print str(confusion_matrix)

def predict(test_data, weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count,img_names,isbest):
    correct=0.0
    confusion_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    counter=1
    if isbest:
        target_file_nn=open('best_output.txt','w')
    if not isbest:
        target_file_nn=open('nnet_output.txt','w')
    for example in test_data:
        input_data = example[0]
        expected_output = example[1]
        hidden_layer_output, main_output = apply_feedforward(input_data, weight_level1, weight_level2,
                                                             hidden_nodes_count)

        #code to
        #this is expected-actual output
        main_index=main_output[0].index(max(main_output[0]))
        actual_index=expected_output.index(1)
        confusion_matrix[actual_index][main_index]+=1
        if main_index==0:
            target_file_nn.write(str(img_names[counter])+" 270\n")
        elif main_index==1:
            target_file_nn.write(str(img_names[counter])+" 180\n")
        elif main_index==2:
            target_file_nn.write(str(img_names[counter])+" 90\n")
        elif main_index==3:
            target_file_nn.write(str(img_names[counter])+" 0\n")
        counter+=1
        if main_index==actual_index:
                # confusion_matrix[actual_index][main_index]+=1
                correct+=1
    target_file_nn.close()
    print "Test Accuracy: " + str(correct/len(test_data) *100)
    print "Confusion Matrix:"
    print('\n'.join([''.join(['{:10}'.format(item) for item in row]) for row in confusion_matrix]))

def initialize_weight(weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count):
    init_matrix(weight_level1,input_node_count,hidden_nodes_count+1,0.0)
    init_matrix(weight_level2,hidden_nodes_count+1,output_node_count,0.0)
    epsilon1 = math.sqrt(6) / math.sqrt(input_node_count + hidden_nodes_count + 1);
    epsilon2 = math.sqrt(6) / math.sqrt(hidden_nodes_count + output_node_count + 1);
    for i in range(input_node_count):
        for j in range((hidden_nodes_count + 1)):
            #weight_level1[i][j] = random.uniform(0, .01)#*2*epsilon1-epsilon1
            weight_level1[i][j] = random.uniform(-.1, .1)
    for i in range((hidden_nodes_count + 1)):
        for j in range(output_node_count):
            weight_level2[i][j] = random.uniform(-.1, .1)


def NN(hidden_count):
    weight_level1 = []
    weight_level2 = []
    train_data = []
    test_data=[]
    img_names=[]
    input_node_count = 193
    hidden_nodes_count = hidden_count
    output_node_count = 4
    #train_file_name="test-data.txt"
    #test_file_name="test-data.txt"
    initialize_weight(weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count)
    print "Weight Initalized successfully"
    read_input(train_file_name, train_data,img_names)
    read_input(test_file_name, test_data,img_names)
    print "Data read successfully"
    train_nn(train_data, weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count,img_names)
    predict(test_data, weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count,img_names,False)
    print "In Confusion Matrix  each row 0,1,2,3 and column 0,1,2,3 corresponds to 270 180 190 0 respectively"

def read_weight(weight_level1,weight_level2):

    for line in open("weight1_output.txt", "r"):
        temp = line.rstrip("\n").split(" ")
        if len(temp)==0:
            continue
        temp=map(float, temp)
        weight_level1.append(temp)
    for line in open("weight2_output.txt", "r"):
        temp = line.rstrip("\n").split(" ")
        if len(temp)==0:
            continue
        temp=map(float, temp)
        weight_level2.append(temp)

def NN_Best(hidden_count):
    weight_level1 = []
    weight_level2 = []
    train_data = []
    test_data=[]
    img_names=[]
    input_node_count = 193
    hidden_nodes_count = hidden_count
    output_node_count = 4
    #train_file_name="test-data.txt"
    #test_file_name="test-data.txt"
    #initialize_weight(weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count)
    print "Weight Initalized successfully"
    read_weight(weight_level1,weight_level2)
    read_input(train_file_name, train_data,img_names)
    read_input(test_file_name, test_data,img_names)
    print "Data read successfully"
    train_nn(train_data, weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count,img_names)
    best=True
    predict(test_data, weight_level1, weight_level2, input_node_count, hidden_nodes_count, output_node_count,img_names,best)
    print "In Confusion Matrix  each row 0,1,2,3 and column 0,1,2,3 corresponds to 270 180 190 0 respectively"

if algorithm == 'knn_modified':
    knn_modified(int(k_hidden_count_best))
if algorithm == 'knn':
    knn(int(k_hidden_count_best))
if algorithm == 'nnet':
    NN(int(k_hidden_count_best))
if algorithm == 'best':
    NN_Best(int(k_hidden_count_best))