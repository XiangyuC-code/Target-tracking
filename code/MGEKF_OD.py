from cmath import inf
from mpmath.functions.functions import _lambertw_special
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.logic.boolalg import distribute_xor_over_and
from sympy.utilities.misc import ARCH
import time

#二维情况下的MGEKF滤波
dT = 1


class MGEKF:
    def __init__(self,init_state,init_P,steps,is_plot=False):
        self.steps = steps
       
        #初始化物体运动
        self.init_state = init_state
        self.s_real = np.mat(np.zeros([4,steps]))    #s_real为物体的真实运动状态
        init_state_err = (np.array(init_state)+np.array([np.random.normal(0,70),np.random.normal(0,250),
                 np.random.normal(0,1),np.random.normal(0,.5)])).reshape(4,1) 
        self.s_real[:,0] = np.mat(init_state_err)
        self.W = np.mat([np.random.normal(0,5,self.steps),np.random.normal(0,5,self.steps),
                 np.random.normal(0,0.1,self.steps),np.random.normal(0,0.1,self.steps)])
    
        self.F = np.mat([[1.,0.,dT,0.],[0.,1.,0.,dT],[0.,0.,1.,0.],[0.,0.,0.,1.]])

        #初始化传感器
        self.z = np.zeros(self.steps)    #z为传感器观测到的数据
        self.o = np.mat(np.zeros([4,steps]))      #o为传感器自身的位置
        self.o[:,0] = np.mat([[0.],[0.],[1.],[0.]])
        self.V = np.random.normal(0,0.001,self.steps)
        self.fai = np.zeros(steps)  #fai为传感器与x轴正方向的夹角

        #初始化卡尔曼滤波器
        self.P_p = np.zeros([steps,4,4])      #P_p为P（k|k-1）
        self.P_u = np.zeros([steps,4,4])      #P_u为P（k|k）
        self.P_p[0,:,:] = np.mat(init_P)
        self.P_u[0,:,:] = np.mat(init_P)
        self.x_p = np.mat(np.zeros([4,steps]))    #x_p为（k|k-1）
        self.x_u = np.mat(np.zeros([4,steps]))      #x_p为（k|k）
        #self.x_p[:,0] = self.s_real[:,0] - self.o[:,0]
        self.x_p[:,0] = np.array(init_state).reshape(4,1) - self.o[:,0] 
        self.x_u[:,0] = self.x_p[:,0]
        self.Q = np.mat([[5,0.,0.,0.],[0.,5,0.,0.],[0.,0.,0.5,0.],[0.,0.,0.,0.1]])
        self.R = np.mat([[0.001]])
        
        self.err_f = False
      
        #求雅克比矩阵
        rx,ry = symbols('rx,ry')
        h = atan(rx/ry)
        self.dh_dx = diff(h,rx)
        self.dh_dy = diff(h,ry)

        self.i = 0
        self.is_plot = is_plot
        #产生物体真实运动轨迹
        self.generate_state()

        self.err_t = 0
        self.inf_nan = False

        self.total_P = 0

        self.num = 0
        

    def generate_state(self):
        #该函数产生物体的真实运动轨迹
        for i in range(self.steps-1):
            self.s_real[:,i+1] = self.F*self.s_real[:,i]+self.W[:,i]#+self.u.T 
        
        
        if self.is_plot:
            self.s_real = np.array(self.s_real)
            plt.figure(1)
            plt.plot(self.s_real[0,:],label='x',color='r')
            plt.plot(self.s_real[1,:],label='y',color='b')
            plt.xlabel('time(s)')
            plt.ylabel('place(m)')
            plt.legend()
            plt.show()
       
            plt.figure(2)
            plt.plot(self.s_real[2,:],label='vx',color='r')
            plt.plot(self.s_real[3,:],label='vy',color='b')
            plt.xlabel('time(s)')
            plt.ylabel('vilocity(m)')
            plt.legend()
            plt.show()
            self.s_real = np.mat(self.s_real)
    
    def get_observation(self,delta):
        #传感器进行一步移动，之后观测物体的位置
        if self.i > 0:
            i = self.i
            #传感器位置更新 速度为1m/s
            self.fai[i] = delta[1]
            self.o[0,i] = self.o[0,i-1] + dT*delta[0]*cos(self.fai[i])
            self.o[1,i] = self.o[1,i-1] + dT*delta[0]*sin(self.fai[i])
            self.o[2,i] = delta[0]*cos(self.fai[i])
            self.o[3,i] = delta[0]*sin(self.fai[i])
            #观测物体位置
            beta = atan((self.s_real[0,i]-self.o[0,i])/(self.s_real[1,i]-self.o[1,i]))
            self.z[i] = beta + self.V[i]
        else:
            beta = atan((self.s_real[0,0]-self.o[0,0])/(self.s_real[1,0]-self.o[1,0]))
            self.z[0] = beta + self.V[0]
            
        
    def filter(self):
        #进行一步卡尔曼滤波
        rx,ry = symbols('rx,ry')
        if self.i > 0:
            i = self.i
            #预测方程
            #b = np.mat([[-self.o[0,i]+self.o[0,i-1]],[-self.o[1,i]+self.o[1,i-1]],[0],[0]])
            #self.x_p[:,i] = self.F*self.x_u[:,i-1] + b 
            b = np.mat([[self.o[0,i]-self.o[0,i-1]-dT*self.o[2,i-1]],[self.o[1,i]-self.o[1,i-1]-dT*self.o[3,i-1]],
                [self.o[2,i]-self.o[2,i-1]],[self.o[3,i]-self.o[3,i-1]]])
            self.x_p[:,i] = self.F*self.x_u[:,i-1] - b + self.W[:,i]
            P = np.mat(self.F*self.P_u[i-1,:,:]*self.F.T+self.Q)
            self.P_p[i,:,:] = P
            
            dhx = self.dh_dx.subs({rx:self.x_p[0,i],ry:self.x_p[1,i]})
            dhy = self.dh_dy.subs({rx:self.x_p[0,i],ry:self.x_p[1,i]})

            H = np.mat([dhx,dhy,0.,0.])
            h_p = atan(self.x_p[0,i]/self.x_p[1,i])

            S = H*P*H.T+self.R
            S = S.astype(np.float)
            #if np.linalg.matrix_rank(S) != 2:
            #    S += np.mat([[.0001,0.],[0.,0.]])
            for m in range(4):
                for n in range(4):
                    if str(P[m,n]) == "inf":
                        self.inf_nan = True
                        break
                    elif str(P[m,n]) == "-inf":
                        self.inf_nan = True
                        break
                    elif str(P[m,n]) == "nan":
                        self.inf_nan = True
                        break
            
            if self.inf_nan == False:
                G = P*H.T*S.I

                #更新方程
                self.x_u[:,i] = self.x_p[:,i]+G*(self.z[i]-h_p)
                bt = self.z[i]
                g = np.mat([cos(bt)/(self.x_p[0,i]*sin(bt)+self.x_p[1,i]*cos(bt)),-sin(bt)/(self.x_p[0,i]*sin(bt)+self.x_p[1,i]*cos(bt)),0.,0.])
                P = np.mat((np.eye(4)-G*g)*P*(np.eye(4)-G*g).T+G*self.R*G.T)
                #P = np.mat((np.eye(4)-G*H)*P) #EKF
                self.P_u[i,:,:] = P

                err = sqrt((self.x_u[0,i] + self.o[0,i] - self.s_real[0,i])**2 + (self.x_u[1,i] + self.o[1,i] - self.s_real[1,i])**2)
                self.err_t += err
                err_his_once[i] = err

                self.total_P += np.trace(self.P_u[i])

                if err >= 1500:
                    self.err_f = True
              
                
    
    def get_err(self):
        #返回平均距离误差
        return self.err_t/(self.i+1)

    def get_Pt(self):
        return self.total_P/(self.i+1)
    
    
    def get_OD(self,delta):
        i = self.i
        if i > 0:
            self.get_observation(delta)

            

            '''
            x = self.s_real[0,i] - self.o[0,i]
            y = self.s_real[1,i] - self.o[1,i]
            xd = self.s_real[2,i] - self.o[2,i]
            yd = self.s_real[3,i] - self.o[3,i]
            '''

            x = self.x_u[0,i-1] + self.o[0,i-1] - self.o[0,i]
            y = self.x_u[1,i-1] + self.o[1,i-1]  - self.o[1,i]
            xd = self.x_u[2,i-1] + self.o[2,i-1]  - self.o[2,i]
            yd = self.x_u[3,i-1] + self.o[3,i-1]  - self.o[3,i]


            r2 = x**2+y**2
            k1 = (xd**2+yd**2)/r2
            k2 = (y*xd-x*yd)/r2

            OD = sqrt((1+k1-np.sqrt((1+k1)**2-4*k2**2))/(1+k1+np.sqrt((1+k1)**2-4*k2**2)))
        else: OD = 0
        return OD
    
    
  

def main():
    #初始化物体状态和方差
    #initial_state = [1000,-4500,10.,-5]
    initial_state = [700,-2500,10.,-5]
    initial_P = np.eye(4)
    
    initial_P[0,0] = 50
    initial_P[1,1] = 100
    initial_P[2,2] = 0.5
    initial_P[3,3] = 0.5
    
    #定义运行时间
    steps = 800
    Filter = MGEKF(initial_state,initial_P,steps)
    Filter.generate_state()

    V = 150

    
    #进行卡尔曼滤波
    for i in range(steps):
        beta = Filter.z[Filter.i-1]
        deltas = [[V,beta+pi/2],[V,beta-pi/2],[V,beta+3*pi/2],[V,beta-3*pi/2]]

        
        OD_list = np.zeros(len(deltas))
        for d in range(len(deltas)):
            if deltas[d][1] >= -pi and deltas[d][1] <= pi:
                OD_list[d] = Filter.get_OD(deltas[d])
            else: OD_list[d] = -1

        sort_result = OD_list.argsort()
        delta = deltas[sort_result[-1]]


        Filter.get_observation(delta)
        Filter.filter()
        
        
        Filter.i += 1
    errm = Filter.get_err()
    return errm


    if False:
        x = Filter.x_u
        s = Filter.s_real
        o = Filter.o
        P_u = Filter.P_u
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
        plt.ylabel('z(m)')
        plt.legend()
        plt.show()

    
        plt.figure(4)
        plt.plot(vy,label='predict',color='b')
        plt.plot(s[3,:],label='real',color='r')
        plt.xlabel('times(s)')
        plt.ylabel('xv(m/s)')
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


if __name__ == "__main__":

    fr = open('err_OD.txt','w')
    ft = open("time_OD.txt",'w')
    error_his = np.zeros(800)

    for l in range(50):
        err_his_once = np.zeros(800)
        start = time.perf_counter()
        err = main()

        fr.write(str(err)+"\n")
        print(err)
        if err <150:
            error_his = error_his + err_his_once

        end = time.perf_counter()
        ft.write(str(end-start)+"\n")

    fr.close()
    ft.close()
    print('done')
    while True:
        pass

'''
for i in range(len(error_his)):
    fte.write(len(error_his[i])+'\n')
'''