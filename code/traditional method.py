from cmath import inf
from re import U
from mpmath.functions.functions import _lambertw_special
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.logic.boolalg import distribute_xor_over_and
from sympy.utilities.misc import ARCH
import math
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
        self.P_p = np.zeros([steps,4,4])      #P_p为P（k|k）
        self.P_u = np.zeros([steps,4,4])      #P_p为P（k|k-1）
        self.P_p[0,:,:] = np.mat(init_P)
        self.P_u[0,:,:] = np.mat(init_P)
        self.x_p = np.mat(np.zeros([4,steps]))    #x_p为（k|k-1）
        self.x_u = np.mat(np.zeros([4,steps]))      #x_p为（k|k）
        #self.x_p[:,0] = self.s_real[:,0] - self.o[:,0]
        self.x_p[:,0] = np.array(init_state).reshape(4,1) - self.o[:,0] 
        self.x_u[:,0] = self.x_p[:,0]
        self.Q = np.mat([[5,0.,0.,0.],[0.,5,0.,0.],[0.,0.,0.5,0.],[0.,0.,0.,0.1]])
        self.R = np.mat([[0.001]])
        self.J = np.zeros([steps,4,4])
        self.J[0,:,:] = np.mat([[70**2,0,0,0],[0,250**2,0,0],[0,0,1**2,0],[0,0,0,.5**2]]).I
        self.J_det = np.zeros(steps)
        self.b = np.zeros(steps)
        self.b[0] = max(sqrt((np.mat(self.J[0,:,:]).I)[0,0]),sqrt((np.mat(self.J[0,:,:]).I)[1,1]))
        #self.J_det[0] = self.J[0,0,0]+self.J[0,1,1]
        self.J_det[0] = np.linalg.det(self.J[0,:,:])
        #self.J_det[0] = np.trace(self.J[0,:,:])
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

    def b_optimal(self,p):
        i = self.i
        U = (p[:,0] + p[:,2])/2
        D = (p[:,1] + p[:,3])/2

        
        rou1 = math.hypot(p[0,0]-self.o[0,i-1],p[1,0]-self.o[1,i-1])
        theta = (atan((p[1,0]-self.o[1,i-1])/(p[0,0]-self.o[0,i-1]))+atan((p[1,1]-self.o[1,i-1])/(p[0,1]-self.o[0,i-1])))/2
        L = np.zeros(2)
        L[0] = rou1*cos(theta) + self.o[0,i-1]
        L[1] = rou1*sin(theta) + self.o[1,i-1]

        rou2 = math.hypot(p[0,2]-self.o[0,i-1],p[1,2]-self.o[1,i-1])
        R = np.zeros(2)
        R[0] = rou2*cos(theta) + self.o[0,i-1]
        R[1] = rou2*sin(theta) + self.o[1,i-1]

        rou3 = math.hypot(U[0]-self.o[0,i-1],U[1]-self.o[1,i-1])
        M = np.zeros(2)
        M[0] = rou3*cos(theta) + self.o[0,i-1]
        M[1] = rou3*sin(theta) + self.o[1,i-1]

        Ub = self.cal_b(U)
        Db = self.cal_b(D)
        Lb = self.cal_b(L)
        Rb = self.cal_b(R)
        p_b = np.zeros(4)
        for j in range(4):
            p_b[j] = self.cal_b(p[:,j])
        
        p_new = np.zeros([2,4])
        b_min = Ub + Lb + p_b[0]
        p_new[:,0] = p[:,0]
        p_new[:,1] = L
        p_new[:,2] = U
        p_new[:,3] = M
        if b_min > Lb + Db + p_b[1]:
            b_min = Lb + Db + p_b[1]
            p_new[:,0] = L
            p_new[:,1] = p[:,1]
            p_new[:,2] = M
            p_new[:,3] = D
        if b_min > Ub + Rb + p_b[2]:
            b_min = Ub + Rb + p_b[2]
            p_new[:,0] = U
            p_new[:,1] = M
            p_new[:,2] = p[:,2]
            p_new[:,3] = R
        if b_min > Rb + Db + p_b[3]:
            b_min = Rb + Db + p_b[3]
            p_new[:,0] = M
            p_new[:,1] = D
            p_new[:,2] = R
            p_new[:,3] = p[:,3]
        
       
        return p_new

    
    def get_observation(self):
        #传感器进行一步移动，之后观测物体的位置
        if self.i > 0:
            i = self.i
            actions = np.array([[10,5*pi/180],[10,-5*pi/180],[100,5*pi/180],[100,-5*pi/180]])
            #传感器位置更新 速度为1m/s
            p = np.zeros([2,4])
            for l in range(4):
                fai = self.fai[i-1] + actions[l,1]
                p[0,l] = self.o[0,i-1] + dT*actions[l,0]*cos(fai)
                p[1,l] = self.o[1,i-1] + dT*actions[l,0]*sin(fai)
            for m in range(12):
                p = self.b_optimal(p)

            self.o[0,i] = (p[0,0]+p[0,1]+p[0,2]+p[0,3])/4
            self.o[1,i] = (p[1,0]+p[1,1]+p[1,2]+p[1,3])/4
            self.o[2,i] = (self.o[0,i] - self.o[1,i-1])/dT
            self.o[3,i] = (self.o[1,i] - self.o[1,i-1])/dT
            self.fai[i] = atan((self.o[1,i]-self.o[1,i-1])/(self.o[0,i]-self.o[0,i-1]))
            self.J[i,:,:] = self.cal_J(self.o[:,i])

            #观测物体位置
            beta = atan((self.s_real[0,i]-self.o[0,i])/(self.s_real[1,i]-self.o[1,i]))
            self.z[i] = beta + self.V[i]
        else:
            beta = atan((self.s_real[0,0]-self.o[0,0])/(self.s_real[1,0]-self.o[1,0]))
            self.z[0] = beta + self.V[0]

    def cal_J(self,o):
        i = self.i
        G1 = (self.s_real[1,i]-o[1])/((self.s_real[0,i]-o[0])**2+(self.s_real[1,i]-o[1])**2)
        G2 = -(self.s_real[0,i]-o[0])/((self.s_real[0,i]-o[0])**2+(self.s_real[1,i]-o[1])**2)
        Gz = np.mat([G1,G2,0,0])
        Jz = Gz.T*Gz/self.R**2
        J = (self.Q+self.F*(np.mat(self.J[i-1,:,:]).I)*(self.F.T)).I + Jz
        return J

    def cal_b(self,o):
        J = self.cal_J(o)
        return max(sqrt(J.I[0,0]),sqrt(J.I[1,1]))
        
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

                if err >= 1500:
                    self.err_f = True
                
    
    
    def get_err(self):
        #返回平均距离误差
        return self.err_t/(self.i+1)

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

    #进行卡尔曼滤波
    for i in range(steps):
        Filter.get_observation()
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
    fe = open("error_tra.txt",'w')
    ft = open("time_tra.txt",'w')
    err_his = np.zeros(800)
    for it in range(20):
        err_his_once = np.zeros(800)
        start = time.perf_counter()
        err = main()
        end = time.perf_counter()
        ft.write(str(end-start)+"\n")
        #print(end-start)


        print('err:',err)
        fe.write(str(err)+"\n")

        if err < 150:
            err_his += err_his_once

    fe.close()
    ft.close()

    print('done')
    while True:
        pass
   

