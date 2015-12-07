__author__ = 'BlackPanther'

data = []

def dataprocessing(fname, data):
    with open(fname) as f:
        for line in f:
            data.append([n if '.jpg' in n else int(n) for n in line.strip().split(' ')])


dataprocessing('temp-test.txt', data)

for each in data:
    print each[1:]