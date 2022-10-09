import tensorflow as tf
import tensorlayer as tl
import random
import numpy as np
from math import pi
from MGEKF_M import *

episode = 5000
batch_size = 32
garmma = 0.9
W = 50
L = 5

def plotfun(agent):
    x = agent.EKF.x_u
    s = agent.EKF.s_real
    o = agent.EKF.o
    P_u = agent.EKF.P_u
    x = np.array(x) 
    s = np.array(s)
    o = np.array(o)

    steps = 800
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
        P[i] = np.trace(P_u[i,:,:])

    plt.figure(1)
    plt.plot(tx,label='predict',color='b')
    plt.plot(s[0,:],label='real',color='r')
    plt.xlabel('times(s)')
    plt.ylabel('x(m)')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(ty,label='predict',color='b')
    plt.plot(s[1,:],label='real',color='r')
    plt.xlabel('times(s)')
    plt.ylabel('y(m)')
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(vx,label='predict',color='b')
    plt.plot(s[2,:],label='real',color='r')
    plt.xlabel('times(s)')
    plt.ylabel('vx(m/s)')
    plt.legend()
    plt.show()

    
    plt.figure(4)
    plt.plot(vy,label='predict',color='b')
    plt.plot(s[3,:],label='real',color='r')
    plt.xlabel('times(s)')
    plt.ylabel('xy(m/s)')
    plt.legend()
    plt.show()
    
    
    plt.figure(5)
    plt.plot(P)
    plt.show()

    plt.figure(6)
    plt.plot(s[0,:],s[1,:],label='real')
    plt.plot(tx,ty,label='predict')
    plt.plot(o[0,:],o[1,:],label='observer')
    plt.show()

class replay_buffer:
    #DQN的经验池
    def __init__(self):
        self.capacity = 10000
        self.buffer = []
        self.position = 0
        

    def push(self,data):
        #将训练数据添加到经验池内
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data
        self.position = int((self.position+1) % self.capacity)
    
    def sample(self,batches_size):
        #在经验池内随机采样
        batch = random.sample(self.buffer,batches_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done 

class Agent:
    def __init__(self,EKF,learning_rate=0.01):
        self.state_dim = 4  #定义状态空间维数
        self.actions = np.array([[10,5*pi/180],[10,-5*pi/180],[-100,5*pi/180],[-100,-5*pi/180],
                        [100,5*pi/180],[100,-5*pi/180]])  #定义动作空间
        self.action_dim = 6     #动作空间维数

        self.model = self.create_network()    #创建训练神经网络
        self.target = self.create_network()      #创建目标神经网络
        self.model.train()
        self.target.eval()
        self.model_optim = tf.optimizers.Adam(lr = learning_rate)       #设置优化器

        #初始化相关状态
        self.state0 = EKF.init_state       
        self.P0 = EKF.P_u[0,:,:]
        self.steps = EKF.steps
        #self.init_state = np.array(EKF.s_real[:,0]).reshape(4,)
        self.EKF = EKF

        self.buffer = replay_buffer()   #创建经验池
        self.epsilon = 0.15
        self.buffer_t = []

        self.err = np.zeros(episode)
        self.r = np.zeros(episode)
        self.P_t = np.zeros(episode)

        self.loss = np.zeros(episode)
        self.total_loss = 0

    def create_network(self):
        #创建神经网络
        input_layer = tl.layers.Input([None,self.state_dim])
        layer1 = tl.layers.Dense(20,act='relu')(input_layer)
        layer2 = tl.layers.Dense(16,act='relu')(layer1)
        output_layer = tl.layers.Dense(self.action_dim)(layer2)
        return tl.models.Model(inputs = input_layer, outputs = output_layer)
    
    def target_update(self):
        #更新目标网络的参数
        for weights,target_weights in zip(self.model.trainable_weights, self.target.trainable_weights):
            target_weights.assign(weights)
    
    def epsilon_greedy(self,state):
        #采用epsilon贪婪策略选择动作
        if random.random() < self.epsilon:
            return random.choice(range(self.action_dim))
        else:
            pred = self.model(np.array([state]))[0]
            return np.argmax(pred)
    
    def replay(self):
        #经验回放，训练神经网络
        self.total_loss = 0
        for i in range(5):
            states,actions,rewards,next_states,dones = self.buffer.sample(batch_size)   #在经验池中随机采样数据，进行训练
            target = self.model(states).numpy()

            next_target = self.target(next_states)
            next_q_value = tf.reduce_max(next_target,axis = 1)    #计算max Q(s',a')

            target[range(batch_size),actions] = rewards + (1-dones)*garmma*next_q_value    #计算y = r + garmma*max Q(s',a')

            with tf.GradientTape() as tape:
                q_pred = self.model(states)
                loss = tf.losses.mean_squared_error(target,q_pred)    #最小化均方误差(y - Q(s,a))^2
            self.total_loss += loss
            grads = tape.gradient(loss,self.model.trainable_weights)
            self.model_optim.apply_gradients(zip(grads,self.model.trainable_weights))
        
        self.target_update()  #训练10次之后更新目标网络的参数
    
    def update_epsilon(self):
        #更新参数epsilon
        self.epsilon *= 0.999
        if self.epsilon <= 0.05:
            self.epsilon = 0.05
    
    def step(self,action):
        #传感器执行一步动作，并返回下一个状态，立即奖励，是否完成该episode
        self.EKF.i += 1
        i = self.EKF.i
        #进行一步卡尔曼滤波
        self.EKF.get_observation(self.actions[action])
        self.EKF.filter()

        next_state = np.array(self.EKF.x_u[:,i]).reshape(4,)  


        #计算立即奖励
        if np.trace(self.EKF.P_u[i]) <= np.trace(self.EKF.P_u[i-1]):
            reward = 1
        else:
            reward = 0
        '''
        range = sqrt((self.EKF.x_u[0,i] + self.EKF.o[0,i] - self.EKF.s_real[0,i])**2 + (self.EKF.x_u[1,i] + self.EKF.o[1,i] - self.EKF.s_real[1,i])**2)
        
        reward = -range/800

        
        u_w0 = 0
        u_wl = 0
        if i <W+L:
            reward = 0
        else:
            for j in range(i-L,i):
                u_w0 += self.EKF.J_det[j]
            for j in range(i-W-L,i-W):
                u_wl += self.EKF.J_det[j]
            if u_w0 >= u_wl:
                reward = 1
            else:
                reward = -1
        '''
        
        Done = False

        return next_state,reward,Done,None

    def training(self):
        for j in range(episode):
            #初始化状态
            self.EKF.__init__(self.state0,self.P0,self.steps)
            self.EKF.i = 0
            total_reward = 0
            Done = 0
            state = np.array(self.EKF.s_real[:,0] - self.EKF.o[:,0],dtype=np.float32).reshape(4,)
            

            for i in range(self.EKF.steps-1):
                action = self.epsilon_greedy(state) #选择动作
                next_state,reward,Done,_ = self.step(action)    #执行动作，返回下一步状态、立即奖励以及是否完成该Episode
                
                if self.EKF.err_f:
                    break

                if self.EKF.inf_nan:
                    break

                total_reward += reward
                state = np.array(state,dtype=np.float32)
                next_state = np.array(next_state,dtype=np.float32)

                if i != 0:
                    self.buffer_t.append((state,action,reward,next_state,Done))
                    

                state = next_state
            
            if self.EKF.inf_nan:
                self.err[j] = self.err[j-1]
                self.r[j] = self.r[j-1]
                self.loss[j] = self.loss[j-1]
                self.P_t[j] = self.P_t[j-1]
                fr.write(str(self.r[j])+"\n")
                fe.write(str(self.err[j])+"\n")
                fp.write(str(self.P_t[j])+"\n")
                fl.write(str(self.loss[j])+"\n")
                print("Found INF or NAN")
            elif self.EKF.err_f:
                self.err[j] = self.err[j-1]
                self.r[j] = self.r[j-1]
                self.loss[j] = self.loss[j-1]
                self.P_t[j] = self.P_t[j-1]
                fr.write(str(self.r[j])+"\n")
                fe.write(str(self.err[j])+"\n")
                fp.write(str(self.P_t[j])+"\n")
                fl.write(str(self.loss[j])+"\n")
                print("Filter error")
            else:
                self.update_epsilon() #更新参数epsilon

                for l in range(len(self.buffer_t)):
                    self.buffer.push(self.buffer_t[l])   #将数据存储到经验池中

                if len(self.buffer.buffer) > batch_size:  #若经验池存储的数据大于一次训练的数据，则开始训练
                    self.replay()      #经验回放，训练神经网络
                
                self.loss[j] = np.mean(self.total_loss)/5
                self.err[j] = self.EKF.get_err()
                self.r[j] = total_reward
                self.P_t[j] = self.EKF.get_Pt()

                fr.write(str(self.r[j])+"\n")
                fe.write(str(self.err[j])+"\n")
                fp.write(str(self.P_t[j])+"\n")
                fl.write(str(self.loss[j])+"\n")

            print('Episode:%d, Reward:%f, Mean error:%.2f, loss:%f, P:%.2f'%(j,self.r[j],self.err[j],self.loss[j],self.P_t[j]))


            
if __name__ == "__main__":
    initial_state = [700,-2500,10.,-5]
    initial_P = np.eye(4)
    initial_P[0,0] = 50
    initial_P[1,1] = 100
    initial_P[2,2] = 0.5
    initial_P[3,3] = 0.5
    steps = 800
    Filter = MGEKF(initial_state,initial_P,steps)
    agent = Agent(Filter)
    fr = open("reward.txt","w")
    fe = open("error.txt","w")
    fp = open("P_u.txt","w")
    fl = open("loss.txt","w")
    agent.training()
    fr.close()
    fe.close()
    fp.close()
    fl.close()
    agent.model.save_weights("model.h5")
    agent.target.save_weights("target.h5")