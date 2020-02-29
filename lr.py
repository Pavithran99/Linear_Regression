from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt


#input and output
x = [0,1,2,3,4,5,6,7,8,9]
y = [1,3,2,5,7,8,8,9,10,12]

learning_rate1 = 0.4
learning_rate = 0.01

teta1 = 10  # initial slope
teta0 = 0 # initial y-intercept

t1=[]
t0=[]
se=[]
plt.ion()
Epoch = 200
for k in range(Epoch):
    y1 = [teta1*i + teta0 for i in x]
    loss = 0
    for i,j in zip(x,y):
        y_cap = teta1*i + teta0
        loss = loss + (y_cap - j)**2
    squared_error = (loss)/(2*len(x))
    print('Epoch: {} Loss: {}'.format(k,squared_error))
    

    sum_derivative_teta0 = 0
    sum_derivative_teta1 = 0
    for i,j in zip(x,y):
        y_cap=teta1*i + teta0
        sum_derivative_teta0 = sum_derivative_teta0 + (y_cap - j)
        sum_derivative_teta1 = sum_derivative_teta1 + ((y_cap - j)*i)

    derivative_teta0 = (sum_derivative_teta0)/len(x)
    derivative_teta1 = (sum_derivative_teta1)/len(x)

    new_teta0 = teta0 - (learning_rate1) * (derivative_teta0)
    new_teta1 = teta1 - (learning_rate) * (derivative_teta1)

    teta0 = new_teta0
    teta1 = new_teta1

    
    #visualize
    if(squared_error<1000):
    	t1.append(teta1)
    	t0.append(teta0)
    	se.append(squared_error)

    fig0=plt.figure(0)
    ax0 = fig0.add_subplot(111)
    ax0.scatter(teta0,squared_error)
    ax0.set_xlabel('teta0')
    ax0.set_ylabel('Mean Squared Error')
    ax0.set_title('Step Size')

    fig1 = plt.figure(1,figsize=(6,6))
    ax1 = fig1.add_subplot(111,projection='3d')
    ax1.plot(t1,t0,se)
    ax1.set_title('Gradient Descent')
    ax1.set_xlabel('teta1')
    ax1.set_ylabel('teta0')
    ax1.set_zlabel('Mean Squared Error')
    
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.plot(x,y1)
    ax2.scatter(x,y)
    ax2.set_xlim([0,10])
    ax2.set_ylim([0,15])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Input vs Output')

    plt.show()
    plt.pause(0.25)
    fig2.clf()
    
