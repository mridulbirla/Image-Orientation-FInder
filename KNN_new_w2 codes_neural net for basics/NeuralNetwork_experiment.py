import random
import math
import numpy as np
#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

class NN:
    LEARNING_RATE = 0.2

    def __init__(self, no_inputs, no_hidden, no_class):
        self.input_vector = no_inputs
        self.hid_lyr = NLayer(no_hidden,1)
        self.op_lyr = NLayer(no_class,1)
        self.level1_weight=self.initializeWeights(no_inputs, no_hidden)
        self.level2_weight=self.initializeWeights(no_hidden, no_class)


    def initializeWeights(self,n_in, n_out):
        ep = math.sqrt(6) / math.sqrt(n_in + n_out + 1)
        W_vector = (np.random.rand(n_out, n_in + 1) * 2 * ep) - ep
        return W_vector

    '''
    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.input_vector))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')
    '''

    def feed_forward(self, inputs):
        aj = self.hid_lyr.apply_feed_forward(inputs,self.level1_weight,True)
        return self.op_lyr.apply_feed_forward(aj,self.level2_weight,False)


    def calculate_deltal(self,actual_op,expected_op):
        expected_op_matrix=np.asmatrix(expected_op)
        return actual_op-expected_op_matrix

    def net_output(self,actual_op):
        actual_op_array=np.asarray(actual_op)
        return actual_op_array*(1-actual_op_array)
        #actual_op_array

    def hidden_multiply(self,total_output_error):
        total_output_error_matrix=np.asmatrix(total_output_error)
        return total_output_error_matrix * self.level2_weight

    def train(self, training_data, iterations):

        for i in range(iterations):
            correct=0.0
            random_list=[]
            random_list.extend(range(0, len(training_data)))
            random.shuffle(random_list)
            for k in random_list:
                OP=self.feed_forward(training_data[k][0])
                p=np.argmax(OP)
                actual_op_index=training_data[k][1].index(1)
                if p==actual_op_index:
                    correct += 1
                deltal_op = self.calculate_deltal(OP,training_data[k][1])
                deltal_hidden=self.net_output(self.hid_lyr.outputs)
                deltal_hidden_list=deltal_hidden.tolist()
                deltal_array=np.asarray(deltal_op)
                total_output_error= deltal_array*self.net_output(OP)
                total_hidden_error=[]
                p=(self.hidden_multiply(total_output_error))
                p_list=np.asarray(p).tolist()
                for l in range(len(deltal_hidden_list[0])):
                    total_hidden_error.append(deltal_hidden_list[0][l] * p_list[0][l])
                #print OP
                self.level1_weight_list=np.asarray(self.level1_weight).tolist()
                hidden_layer_op=self.hid_lyr.outputs.tolist()
                total_output_error_list=total_output_error.tolist()
                self.level2_weight_list=np.asarray(self.level2_weight).tolist()
                self.level1_weight_list = self.level1_weight.tolist()
                for output_node_no in range(len(self.level2_weight_list)):
                    for hidden_node_no in range(len(self.level2_weight_list[output_node_no])):
                        self.level2_weight_list[output_node_no][hidden_node_no]+=(self.LEARNING_RATE*total_output_error_list[0][output_node_no]*hidden_layer_op[0][hidden_node_no])
                for hidden_node_no in range(len(self.level1_weight_list)):
                    for input_no in range(len(self.level1_weight_list[hidden_node_no])):
                        self.level1_weight_list[hidden_node_no][input_no]+=(self.LEARNING_RATE * total_hidden_error[hidden_node_no] *training_data[k][0][input_no])
                self.level1_weight=np.asarray(self.level1_weight_list)
                self.level2_weight=np.asarray(self.level2_weight_list)
            print "Accuracy=" + str(correct/len(training_data)*100)+ " for Iteration "+ str(i)
class NLayer:

    def __init__(self, neurons_no, bias):
        self.bias = bias if bias else random.random()
        self.outputs=np.ndarray([])
        self.nodes = []
        self.no_of_nodes=neurons_no
        #[self.nodes.append(Node(self.bias)) for i in range(num_neurons)]

    def apply_feed_forward(self, inputs,weight,flag_add_bias):
        weight_matrix=np.asmatrix(weight)
        weight_matrix_transpose=weight_matrix.T
        input_matrix=np.matrix(inputs)
        aj=np.dot(input_matrix,weight_matrix_transpose)
        zj = self.sigmoid(aj)
        self.outputs=zj
        if flag_add_bias==True:
            self.outputs = np.append(self.outputs,np.ones([len(self.outputs),1]),1)
        return self.outputs

    def get_outputs(self):
        pass

    def sigmoid(self,z):

        #1.026187963 -10
        #print('sig,oid part 1')
        x=np.divide(1.0,(1.0+np.exp(-z)))
        return  x


def read_input(file_name,input_data):
    for line in open(file_name, "r"):
        temp = line.rstrip("\n").split(" ")
        p=[0,0,0,0]
        # 0 => [0,0,0,1]   90 => [0,0,1,0] 180 = [0,1,0,0] 270 = [1,0,0,0]     p=[0,0,0,1]
        if int(temp[1])==0:
            p[3]=1
        elif int(temp[1])==90:
            p[2]=1
        elif int(temp[1])==180:
            p[1]=1
        elif int(temp[1])==270:
            p[0]=1
        t = map(int, temp[2:])
        #for i in range(0,len(t)):
        #    t[i]/=255.00
        t.append(1.0)
        #map((lambda x: x/255.0), t)
        q=[t,p]
        input_data.append(q)


hidden_node=30
input_node=192
class_node=4

#initial_w1 = initializeWeights(input_node, hidden_node);
#initial_w2 = initializeWeights(hidden_node, class_node);
#p=initial_w1.flatten()
#q=initial_w2.flatten()
#initial_w1_list=p.tolist()
#initial_w2_list=q.tolist()

nn = NN(input_node, hidden_node, class_node)
train_data=[]

read_input("train-data-small.txt",train_data)
nn.train(train_data,70)
# XOR example:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]

# nn = NN(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))
