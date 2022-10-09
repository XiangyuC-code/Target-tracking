import numpy as np
import matplotlib.pyplot as plt
from MGEKF_M import *
import tensorflow as tf
import tensorlayer as tl
import time


reward_plot = False
error_plot = False
loss_plot = False
P_plot = False
tracking_plot = True



def greedy(state):
    pred = model(np.array([state]))[0]
    return np.argmax(pred)

def create_network():
    #创建神经网络
    input_layer = tl.layers.Input([None,4])
    layer1 = tl.layers.Dense(20,act='relu')(input_layer)
    layer2 = tl.layers.Dense(16,act='relu')(layer1)
    output_layer = tl.layers.Dense(6)(layer2)
    return tl.models.Model(inputs = input_layer, outputs = output_layer)


if tracking_plot:

    for it in range(20):

        initial_state = [700,-2500,10.,-5]
        initial_P = np.eye(4)
        initial_P[0,0] = 50
        initial_P[1,1] = 100
        initial_P[2,2] = 0.5
        initial_P[3,3] = 0.5
        steps = 800
        Filter = MGEKF(initial_state,initial_P,steps)
        
        actions = np.array([[10,5*pi/180],[10,-5*pi/180],[-100,5*pi/180],[-100,-5*pi/180],
                            [100,5*pi/180],[100,-5*pi/180]])  #定义动作空间

        model = create_network()
        model.load_weights('model.h5')
        model.eval()

        state = np.array(Filter.s_real[:,0] - Filter.o[:,0],dtype=np.float32).reshape(4,)
        for i in range(steps):
            action = greedy(state) #选择动作
            #进行一步卡尔曼滤波
            Filter.get_observation(actions[action])
            Filter.filter()
            Filter.i += 1

        err = Filter.get_err()
        print(err)

        x = Filter.x_u
        s = Filter.s_real
        o = Filter.o
        
        x = np.array(x) 
        s = np.array(s)
        o = np.array(o)

        tx = np.zeros(steps)
        ty = np.zeros(steps)
        vx = np.zeros(steps)
        vy = np.zeros(steps) 
        P = np.zeros(steps)
        for i in range(steps):
            tx[i] = x[0,i]+o[0,i]
            ty[i] = x[1,i]+o[1,i]
            vx[i] = x[2,i]+o[2,i]
            vy[i] = x[3,i]+o[3,i]

        plt.figure(1)
        plt.plot(tx,label='predict',color='b')
        plt.plot(s[0,:],label='real',color='r')
        plt.xlabel('time(s)')
        plt.ylabel('x(m)')
        plt.legend()
        plt.show()

        plt.figure(2)
        plt.plot(ty,label='predict',color='b')
        plt.plot(s[1,:],label='real',color='r')
        plt.xlabel('time(s)')
        plt.ylabel('y(m)')
        plt.legend()
        plt.show()

        plt.figure(3)
        plt.plot(vx,label='predict',color='b')
        plt.plot(s[2,:],label='real',color='r')
        plt.xlabel('time(s)')
        plt.ylabel('vx(m/s)')
        plt.legend()
        plt.show()

        
        plt.figure(4)
        plt.plot(vy,label='predict',color='b')
        plt.plot(s[3,:],label='real',color='r')
        plt.xlabel('time(s)')
        plt.ylabel('vy(m/s)')
        plt.legend()
        plt.show()
        

        plt.figure(5)
        plt.plot(s[0,:],s[1,:],label='real')
        plt.plot(tx,ty,label='predict')
        plt.plot(o[0,:],o[1,:],label='observer')
        plt.show()




if reward_plot:
    r = []
    fr = open("reward.txt")               # 返回一个文件对象   
    line = fr.readline()           # 调用文件的 readline()方法   
    while line:
        r.append(float(line))     
        line = fr.readline()
    fr.close()

    r_p = np.zeros((len(r)//100+1))
    for i in range(len(r)):
        r_p[i//100] +=  r[i] 
    r_p /= 100

    plt.plot(r_p[7:-1])
    plt.xlabel('Training Iteration (×100)')
    plt.ylabel('Average Reward')
    plt.ylim(220,300)
    plt.xticks(range(0,17,2))
    plt.show()

if error_plot:
    e = []
    fe = open("error.txt")               # 返回一个文件对象   
    line = fe.readline()           # 调用文件的 readline()方法   
    while line:
        e.append(float(line))     
        line = fe.readline()
    fe.close()

    e_p = np.zeros((len(e)//100+1))
    for i in range(len(e)):
        e_p[i//100] +=  e[i]**2
    e_p /= 100
    e_p = np.sqrt(e_p)

    plt.plot(e_p[7:-1])
    plt.xlabel('Training Iteration (×100)')
    plt.ylabel('Average Location RMSE(m)')
    plt.xticks(range(0,17,2))
    plt.show()


if loss_plot:
    l = []
    fl = open("loss.txt")               # 返回一个文件对象   
    line = fl.readline()           # 调用文件的 readline()方法   
    while line:
        l.append(float(line))     
        line = fl.readline()
    fl.close()

    l_p = np.zeros((len(l)//100+1))
    for i in range(len(l)):
        l_p[i//100] +=  l[i]
    l_p /= 100

    plt.plot(l_p[7:-1])
    plt.xlabel('Training Iteration (×100)')
    plt.ylabel('Loss')
    plt.xticks(range(0,17,2))
    plt.show()

if P_plot:
    p = []
    fp = open("P_u.txt")               # 返回一个文件对象   
    line = fp.readline()           # 调用文件的 readline()方法   
    while line:
        p.append(float(line))     
        line = fp.readline()
    fp.close()

    p_p = np.zeros((len(p)//100+1))
    for i in range(len(p)):
        p_p[i//100] +=  p[i]
    p_p /= 400*800

    plt.plot(p_p[7:-1])
    plt.xlabel('Training Iteration (×100)')
    plt.ylabel('P')
    plt.xticks(range(0,17,2))
    plt.show()
