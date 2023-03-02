import numpy as np
import torch
import torch.nn as nn
import math
import data_prepare
import scipy.linalg
import torch.nn.functional as F

class EAFNN(nn.Module):
    def __init__(self,ruler_num=5,input_num=2,x_num=1,y_num=1):
        super(EAFNN, self).__init__()
        self.L=ruler_num
        self.x_num=x_num
        self.y_num=y_num
        self.input_num=input_num
        self.c = nn.Parameter(torch.Tensor(input_num, ruler_num))
        self.b = nn.Parameter(torch.Tensor(input_num, ruler_num))
        self.w = nn.Parameter(torch.Tensor(ruler_num,x_num,y_num))   

        nn.init.xavier_uniform_(self.c)
        nn.init.xavier_uniform_(self.b)
        nn.init.xavier_uniform_(self.w)


    def forward(self, input, x):
        num_data,num_feature=input.shape
        input = torch.transpose(input,0,1)
        x = torch.transpose(x,0,1).float()


        diyiceng=torch.ones([num_feature*self.L,num_data])
        for i in range(num_feature):
            for j in range(self.L):
                diyiceng[i*self.L+j,:]=torch.exp(-(input[i,:]-self.c[i,j])**2/(2*(self.b[i,j]**2)))
        dierceng=torch.ones([self.L,num_data])
        for i in range(self.L):
            temp=diyiceng[i:num_feature*self.L:self.L,:]
            dierceng[i,:] = torch.prod(temp,axis=0)+0.001
        lsdsum=torch.sum(dierceng,0)
        #lsdsum=1
        disanceng=torch.div(dierceng,lsdsum)

        disiceng=torch.ones([self.L,self.y_num,num_data])
        for i in range(self.L):
            disiceng[i,:,:]=torch.mm(torch.transpose(self.w[i,:,:],0,1),x)
        diwuceng = torch.ones([self.y_num, num_data])
        for i in range(num_data):
            diwuceng[:,i] = torch.mm(disanceng[:,i].resize(1,self.L),disiceng[:,:,i])
        return disanceng,diwuceng
    
    def addrule(self,i):

        c_new1 = self.c[:,i] + 1/6*self.b[:,i]
        c_new1 = c_new1.reshape(self.input_num,1)
        c_new2 = self.c[:,i] - 1/6*self.b[:,i]
        c_new2 = c_new2.reshape(self.input_num,1)
        b_new1 = 2/3*self.b[:,i]
        b_new1 = b_new1.reshape(self.input_num,1)
        b_new2 = 2/3*self.b[:,i]
        b_new2 = b_new2.reshape(self.input_num,1)
        
        
        input_num=self.input_num
        ruler_num=self.L+1
        y_num=self.y_num
        x_num= self.x_num
        
        c=self.c.data
        b=self.b.data
        w=self.w.data

        
        self.c = nn.Parameter(torch.Tensor(input_num, ruler_num))
        self.b = nn.Parameter(torch.Tensor(input_num, ruler_num))
        self.w = nn.Parameter(torch.Tensor(ruler_num,x_num,y_num))   

        
        self.c.data = torch.cat((c[:,0:i],c_new1,c_new2,c[:,i+1:self.L]),1)
        self.b.data = torch.cat((b[:,0:i],b_new1,b_new2,b[:,i+1:self.L]),1)
        self.w.data = torch.cat((w[0:i,:,:],w[i:i+1,:,:],w[i:i+1,:,:],w[i+1:self.L,:,:]),0)
        self.L=self.L+1
        
    def subrule(self,i):
        input_num=self.input_num
        ruler_num=self.L-1
        y_num=self.y_num
        x_num= self.x_num
        
        c=self.c.data
        b=self.b.data
        w=self.w.data

                
        
        self.c = nn.Parameter(torch.Tensor(input_num, ruler_num))
        self.b = nn.Parameter(torch.Tensor(input_num, ruler_num))
        self.w = nn.Parameter(torch.Tensor(ruler_num,x_num,y_num))   

        
        self.c.data = torch.cat((c[:,0:i],c[:,i+1:self.L]),1)
        self.b.data = torch.cat((b[:,0:i],b[:,i+1:self.L]),1)
        self.w.data = torch.cat((w[0:i,:,:],w[i+1:self.L,:,:]),0)
        self.L=self.L-1
        
def train_epoch(epoch, model, dataloaders):
    LEARNING_RATE = 0.005
    #LEARNING_RATE = 1*e**(-epoch/100)
    print('learning rate{: .4f}'.format(LEARNING_RATE))

    optimizer = torch.optim.SGD([
        {'params': model.c,'lr': LEARNING_RATE/10},
        {'params': model.b, 'lr': LEARNING_RATE/10},
        {'params': model.w, 'lr': LEARNING_RATE},

    ], lr=LEARNING_RATE / 10)

    model.train()

    iter_data = iter(dataloaders)

    num_iter = len(iter_data)
    for i in range(1, num_iter):
        data, label = iter_data.next()
        data, label = data, label.float()
        label = torch.transpose(label,0,1)


        optimizer.zero_grad()
        input = data[:,0:2]
        disanceng,label_pred= model(input,data)
        
        for i in range(model.L):
            if torch.mean(disanceng[i,:])>0.3:
                model.addrule(i)       
                break
            if torch.mean(disanceng[i,:])<0.01:
                model.subrule(i)
                break
        
        disanceng,label_pred= model(input,data)
        loss = F.mse_loss(label_pred.float(),label)
        loss.backward()
        optimizer.step()

def EAFNN_MPC(model,input,x,N_p,N_c,n,m,lamuda,y_set,err):
    disanceng,diweceng=model(input,x)
    y_num=model.y_num
    finalmodel=torch.matmul(model.w.T, disanceng).squeeze(dim=2)
    W_y=finalmodel[0,0:n].detach().numpy()
    W_x=finalmodel[0,n:].detach().numpy()
    tempA = np.ones([1,y_num])
    TEMPA = np.empty([0,0])
    for i in range(n):
        TEMPA = scipy.linalg.block_diag(TEMPA,tempA)
    A = np.fliplr(TEMPA)
    for i in range(N_p):
        k=i+n
        a=0
        for j in range(n):
            a=a+W_y[j]*A[k-j-1,:]
        A = np.insert(A, k, values=a, axis=0)
    A=A[n:,:]
    
    Bs=W_x[1:]
    Bs=Bs.reshape(1,m-1)
    tempBs=W_x[1:]
    for i in range(N_p-1):
        k=i+1
        tempBs=np.append(tempBs[1:],0)
        Bs = np.insert(Bs, k, values=tempBs, axis=0)
        
    B = np.zeros([n,m-1])
    for i in range(N_p):
        k=i+n
        b=np.array([0])
        for j in range(n):
            b=b+W_y[j]*B[k-j-1,:]
        B = np.insert(B, k, values=b+Bs[i,:], axis=0)
    B=B[n:,:]

    Cs=np.empty([0,N_p])
    tempCs=np.zeros([N_p])
    for i in range(N_p):
        k=i
        if i>=m:
            tempCs=np.append(0,tempCs[0:N_p-1])
        else:
            tempCs=np.append(W_x[i],tempCs[0:N_p-1])
        Cs = np.insert(Cs, k, values=tempCs, axis=0)
        
    C = np.zeros([n,N_p])
    for i in range(N_p):
        k=i+n
        c=np.array([0])
        for j in range(n):
            c=c+W_y[j]*C[k-j-1,:]
        C = np.insert(C, k, values=c+Cs[i,:], axis=0)
    C=C[n:,:]
    tempClast=0
    for i in range(N_c-1,N_p):
        tempClast=tempClast+C[:,i]
    C=C[:,0:N_c-1]
    C = np.insert(C, N_c-1, values=tempClast, axis=1)
    
    E = np.zeros([n,1])
    bias=0
    bias=bias+err
    for i in range(N_p):
        k=i+n
        e=np.array([0])
        for j in range(n):
            e=e+W_y[j]*E[k-j-1,:]
        E = np.insert(E, k, values=e+bias, axis=0)
    E=E[n:,:]
      
    Yset=y_set*np.ones([N_p,1])
    Yl=x[0,0:n].detach().numpy().reshape(n,1)
    Ul=x[0,n:-1].detach().numpy().reshape(m-1,1)
    uk=x[0,n]*np.ones([N_c,1])
    uk=uk.detach().numpy()
    Error=Yset-np.matmul(A,Yl)-np.matmul(B,Ul)-E-np.matmul(C,uk)
    K=np.ones([N_c,N_c])
    K=np.tril(K)
    
    detaU = np.matmul(K.T,C.T)
    detaU = np.matmul(detaU,C)
    detaU = np.matmul(detaU,K)
    detaU=np.linalg.inv(detaU+lamuda*np.eye(N_c))   
    detaU = np.matmul(detaU,K.T)
    detaU = np.matmul(detaU,C.T)
    detaU=np.matmul(detaU,Error)
    return detaU[0,0],diweceng