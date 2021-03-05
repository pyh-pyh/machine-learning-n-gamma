import torch


def get_data(path):
    with open(path, 'r') as f:
        data = f.readlines()[3:]
        data = [float(data[i].strip()) for i in range(len(data))]
    return data


def read_data(din, dout, d1, d2, l1, l2):
    data = []
    for i in range(d1):
        data.append(get_data(r'D:\\My Files\\核技术专题研讨\\n-y data\\n-y data\\neutron\\' + str(i+1) + '.csv'))
    for i in range(d2):
        data.append(get_data(r'D:\\My Files\\核技术专题研讨\\n-y data\\n-y data\\gamma\\' + str(i+1) + '.csv'))
    data1 = data[:d1]
    data2 = data[d1:]

    indata = torch.empty(l1+l2, din)
    for i in range(l1):
        for j in range(din):
            indata[i, j] = data1[i][j]
    for i in range(l2):
        for j in range(din):
            indata[i+l1, j] = data2[i][j]
    
    outdata = torch.empty(l1+l2, dout)
    for i in range(l1):
        outdata[i, 0] = 0
        outdata[i, 1] = 1
    for i in range(l2):
        outdata[i+l1, 0] = 1
        outdata[i+l1, 1] = 0

    checkdata = torch.tensor(data1[l1:] + data2[l2:])
    label = torch.tensor([[0, 1] for i in range(len(data1[l1:]))] + [[1, 0] for i in range(len(data2[l2:]))])

    return [indata, outdata, checkdata, label]


def train(N, din, H, dout, it):
    indata = read_data(din, dout, 33, 28, 20, 20)[0]
    outdata = read_data(din, dout, 33, 28, 20, 20)[1]
    
    model = torch.nn.Sequential(
        torch.nn.Linear(din, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, dout),
    )
    
    lossfunc = torch.nn.MSELoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for i in range(it):
        predy = model(indata)

        loss = lossfunc(predy, outdata)
        print(i+1, loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    check(model, din, dout)


def check(model, din, dout):
    checkdata = read_data(din, dout, 33, 28, 20, 20)[2]
    label = read_data(din, dout, 33, 28, 20, 20)[3]
    lossfunc = torch.nn.MSELoss()
    loss = lossfunc(model(checkdata), label)
    print('loss = ', loss.item())

    result = []
    correct = 0
    for i in range(len(checkdata)):
        if (model(checkdata)[i][0] > 0.5 and model(checkdata)[i][1] < 0.5) or (model(checkdata)[i][0] > model(checkdata)[i][1]):
            result.append([1, 0])
        elif (model(checkdata)[i][1] > 0.5 and model(checkdata)[i][0] < 0.5) or (model(checkdata)[i][1] > model(checkdata)[i][0]):
            result.append([0, 1])
        print(i+1, model(checkdata)[i], result[i], list(label[i]))
        if result[i] == list(label[i]):
            correct += 1
    corate = correct / len(checkdata)
    print('Accuracy = ', 100*corate, '%')


train(40, 252, 100, 2, 20000)