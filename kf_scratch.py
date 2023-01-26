# Written by Quang Nhat Le
# Please feel free to use it.
# quangle@umich.edu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

class KalmanFilter():

    # def __init__(self, arguments):
    #     # define or assign object attributes
        
    # def other_methods(self, arguments):
    #     # body of the method
    
    # INPUT
    # A, B : state space matrices (nxn nxm)
    # C : measurement matrix (pxn)
    # R : cov matrix for state noise (nxn)
    # Q : cov matrix for measurement noise (pxp)
    # S0 : Initial cov matrix for state (nxn)
    # x0 : Initial state 

    # OUTPUT
    # S, x : new state and new cov matrix for state (nxn and n)
    def __init__(self,A = None, B = None,C = None, Q =None, R = None, S =None, x0 = None,u =None):
        
        if(A is None or C is None):
            raise ValueError("System dynamics should be set.")

        # attributes
        self.A = A
        self.C = C
        self.B = B

        # shapes
        self.n = A.shape[0]
        self.p = C.shape[0]
        self.m = B.shape[1]

        self.Q = np.eye(self.p) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.S = np.eye(self.n) if S is None else S
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
        self.u = np.zeros((self.m,1)) if u is None else u

    def predict(self):
        # mean
        self.x = np.matmul(self.A,self.x) + np.matmul(self.B,self.u)
        
        # cov matrix
        self.S = np.matmul(self.A,np.matmul(self.S,np.transpose(self.A))) + self.R

    
    def update(self,y):

        # call predictions before update
        self.predict()

        # Kalman Gain
        K = np.matmul(self.S,np.matmul(np.transpose(self.C),    
            np.linalg.inv(np.matmul(self.C,np.matmul(self.S,
            np.transpose(self.C))) + self.Q)))
        
        # new state
        self.x = self.x + np.matmul(K,(y - np.matmul(self.C,self.x)))

        # new cov matrix
        self.S = np.matmul(np.identity(self.n) - np.matmul(K,self.C),self.S)

        return self.x, self.S


def legend_without_duplicate_labels(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc='lower right')

def example():
    Y = np.genfromtxt('test_data.txt')[:,1]

    plt.figure(figsize=(18,6))
    # plt.plot(np.arange(len(Y)), np.log(Y))

    Q = np.array([[10]]) # measurement covariance matrix
    C = np.array([[1,0]]) # observartion matrix
    A = np.array([[1,1],[0,1]]) # state transition matrix
    R = np.array([[100,0],[0,100]]) # process noise matrix

    B = np.zeros((2,1))


    # initial guess of state and estimate uncertainty
    X_0 = np.hstack([[0],[0]])
    P_0 = np.array([[1000,0],[0,1000]])

    # define object
    kf = KalmanFilter(A=A,B=B,C=C,Q=Q,R=R,x0=X_0,S=P_0)
    predictions = []
    
    log_y = np.log(Y)
    # time loop
    for z in log_y:
        # predictions.append(np.dot(C,  kf.predict())[0])
        # kf.update(z)
        pred,_ = kf.update(z)
        predictions.append(np.matmul(C,pred))


    # show result
    plt.plot(range(len(log_y)), log_y, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    # plt.legend()
    legend_without_duplicate_labels(plt)
    plt.xlim([90,150])
    plt.ylim([0,2])
    plt.show()

def robot_estimate():
    data_1 = pd.read_csv("icart_and_human_corridor_5.csv",header = None,index_col=None)

    #for robot
    x1 = data_1.iloc[:, 2].to_numpy()
    y1 = data_1.iloc[:, 3].to_numpy()

    training_Y = np.vstack((x1,y1)).T
    A = np.identity(2) # state transition matrix
    B = np.zeros((2,1))
    C = np.identity(2) # observartion matrix
    R = np.array([[10,9],[9,10]]) # process noise matrix
    Q = np.array([[10000,1000],[1000,10000]]) # measurement covariance matrix

    # define object
    kf = KalmanFilter(A=A,B=B,C=C,Q=Q,R=R)
    predictions = np.zeros((2,0))

    for z in training_Y:
        pred,_ = kf.update(z.reshape(-1,1))
        predictions = np.concatenate((predictions,np.matmul(C,pred)),axis=1)
        # predictions = np.concatenate((predictions,np.matmul(C,  kf.predict())),axis=1)
        # kf.update(z.reshape(-1,1))

    plt.plot(x1, y1, label = 'Measurements')
    plt.plot(predictions[0,:],predictions[1,:], label = 'Kalman Filter Prediction')
    plt.legend()
    plt.show()


def main():
    dt = 1.0/60

    # 3x3 , state-to-state 
    A = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]]) 
    
    # 1x3 p = 1, n = 3 , state-to-measurement
    C = np.array([1, 0, 0]).reshape(1, 3)

    # n = 3, m = 2 , input-to-state
    B = np.zeros((3,2))

    # input
    u = np.zeros((2,1))
    
    # cov matrices
    R = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    Q = np.array([0.5]).reshape(1, 1)

    # example measurements and state
    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

    # define object
    kf = KalmanFilter(A=A,B=B,C=C,Q=Q,R=R,u=u)
    predictions = []

    # time loop
    for z in measurements:
        predictions.append(np.dot(C,  kf.predict())[0])
        kf.update(z)

    # show result
    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main()
    example()
    # robot_estimate()