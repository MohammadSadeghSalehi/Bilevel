#HOAG Algorithm
import scipy.sparse as sp
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

class HOAG:
    def __init__(self, A, b, y, x0, L, mu, alpha,nu, xi, max_iter=2000, tol=1e-8, lowersolver='FISTA'):
        # A is the forward operator
        #b is the ground truth
        # y is the noisy image
        # x0 is the initial guess for low-level problem
        # L is the lipschitz constant
        # mu is the strong-convexity parameter
        # nu is the smoothing parameter for TV
        
        self.A = A
        self.b = b
        self.y = np.squeeze(np.asarray(y))
        self.matsize = np.shape(x0)
        self.x0 = np.zeros((np.shape(b)[2],len(x0.flatten())))
        self.x0[:,:] = x0.flatten()
        self.L = L
        self.mu = mu
        self.nu = nu
        self.xi = xi
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.gtol = 0
        self.upperObjective = []
        self.lowerIter = []
        self.lowerIterSum = 0
        self.lowersolver = lowersolver
    def FISTA(self, alpha,nu,xi,i,epsilon):
        # Algorithm for solving the lower-level problem
        x = self.x0[i,:]

        t = 0
        for k in range(self.max_iter):
            x_old = x
            x = np.reshape(x,(256,256))
            x = x.flatten()
            self.L = 1+ (np.exp(alpha)/np.exp(nu))* 8+np.exp(xi)
            self.mu = 1+np.exp(xi)
            tau = 1/(self.L)
            q = tau*self.mu
            t_old = t
            t = (1-q*t**2+np.sqrt((1-q*t**2)**2+4*t**2))/2
            beta = ((t_old-1)*(1-t*q))/(t*(1-q))
            z = x + beta * (x-x_old)
            p = self.gradPhi(z, alpha, nu, xi,i)
            if np.linalg.norm(p)**2/self.mu**2 < epsilon:
                self.x0[i,:] = z -  tau*p
                self.lowerIterSum += k
                print("i = "+ str(i)+ " Converged at iteration: " + str(k+1))
                
                return self.x0[i,:]
            x = z -  tau*p
        self.x0[i,:] = x
        return x
    def phi(self, x, alpha,nu,xi, i):
        phi = 0.5*np.linalg.norm(x.flatten()-self.y[:,:,i].flatten())**2 +np.exp(alpha) * self.TV2D(x, np.exp(nu)) + 0.5 *np.exp(xi) * np.linalg.norm(x)**2
        return phi
    def gradPhi(self, x, alpha,nu,xi,i):
        x = np.reshape(x,self.matsize)

        gradPhi = np.squeeze(np.asarray(x.flatten() - np.squeeze(np.asarray(self.y[:,:,i].flatten())) + np.exp(alpha) *self.gradTV2D(x, np.exp(nu)) + np.exp(xi) * x.flatten()))
        return gradPhi.flatten()
    def hessianPhi(self, x, alpha,nu,xi,i,d):
        x = np.reshape(x,self.matsize)
        d = np.reshape(d,self.matsize)
        hess = (1+np.exp(xi))*d + np.exp(alpha) * self.hessianTV2D(x, np.exp(nu),d) 
        return hess
    def TV(self,x, nu):
        return (np.sum(np.sqrt(np.abs(x[1:]-x[:-1])**2+ nu**2)))
    def TV2D(self,x, nu):
        x = np.reshape(x,(256,256))
        tv = 0
        tv += (np.sum(np.sqrt(np.abs(x[:,1:]-x[:,:-1])**2+ nu**2)))
        tv += (np.sum(np.sqrt(np.abs(x[1:,:]-x[:-1,:])**2+ nu**2)))
        return tv
    
    def gradTV2D(self,x, nu):

        x = np.reshape(x,(self.matsize[0],self.matsize[1]))

        grad = np.zeros(np.shape(x))
        grad[:,:-1]+= -(x[:,1:]-x[:,:-1])/np.sqrt(np.abs(x[:,1:]-x[:,:-1])**2+ nu**2)
        grad[:,1:] += (x[:,1:]-x[:,:-1])/np.sqrt(np.abs(x[:,1:]-x[:,:-1])**2+ nu**2) 
          
        grad[:-1,:]+= -(x[1:,:]-x[:-1,:])/np.sqrt(np.abs(x[1:,:]-x[:-1,:])**2+ nu**2)
        grad[1:,:] += (x[1:,:]-x[:-1,:])/np.sqrt(np.abs(x[1:,:]-x[:-1,:])**2+ nu**2)             
        return grad.flatten()
    def hessianTV2D(self,x, nu,d):
        hess = np.zeros(self.matsize)
        left = np.zeros(self.matsize)
        right = np.zeros(self.matsize)
        up = np.zeros(self.matsize)
        down = np.zeros(self.matsize)

        #Matrix-vector version
        down[:,:-1] = 1-(x[:,1:]-x[:,:-1])**2/np.sqrt(np.abs(x[:,1:]-x[:,:-1])**2+ nu**2)
        hess[:,:-1] += np.multiply(-down[:,:-1] , d[:,:-1])
        up[:,1:] = 1-(x[:,1:]-x[:,:-1])**2/np.sqrt(np.abs(x[:,1:]-x[:,:-1])**2+ nu**2) 
        hess[:,1:] += np.multiply(-up[:,1:] , d[:,1:])
        right[:-1,:] = 1-(x[1:,:]-x[:-1,:])**2/np.sqrt(np.abs(x[1:,:]-x[:-1,:])**2+ nu**2)
        hess[:-1,:]+= np.multiply(-right[:-1,:] , d[:-1,:])
        left[1:,:] = 1-(x[1:,:]-x[:-1,:])**2/np.sqrt(np.abs(x[1:,:]-x[:-1,:])**2+ nu**2)
        hess[1:,:] += np.multiply(-left[1:,:] , d[1:,:])
        hess[:,:] += d[:,:] @ (left + right + up + down)
        
        return hess
    def gradTV(self,x, nu):
        grad = np.zeros(len(x))
        grad[:-1]+= -(x[1:]-x[:-1])/np.sqrt(np.abs(x[1:]-x[:-1])**2+ nu**2)
        grad[1:] += (x[1:]-x[:-1])/np.sqrt(np.abs(x[1:]-x[:-1])**2+ nu**2)
        
        return np.squeeze(np.asarray(grad))    
    def partialNuGradTV(self,x, nu):
        x = np.reshape(x,(self.matsize[0],self.matsize[1]))
        grad = np.zeros(np.shape(x))
        grad[:,:-1]+= nu *(x[:,1:]-x[:,:-1])/np.sqrt(np.abs(x[:,1:]-x[:,:-1])**2+ nu**2)
        grad[:,1:] += -nu*(x[:,1:]-x[:,:-1])/np.sqrt(np.abs(x[:,1:]-x[:,:-1])**2+ nu**2) 
          
        grad[:-1,:]+= nu*(x[1:,:]-x[:-1,:])/np.sqrt(np.abs(x[1:,:]-x[:-1,:])**2+ nu**2)
        grad[1:,:] += -nu*(x[1:,:]-x[:-1,:])/np.sqrt(np.abs(x[1:,:]-x[:-1,:])**2+ nu**2)             
        return grad 

    def CG(self,alpha,nu,xi,i,x,b,tol):
        r = b-self.hessianPhi(x,alpha,nu,xi,i,x).flatten()
        p = r
        rsold = np.dot(r.T,r)
        for k in range(100):
            Ap = self.hessianPhi(x,alpha,nu,xi,i,p).flatten()
            alpha = rsold/np.dot(p.T,Ap)
            x = x + alpha*p
            r = r - alpha*Ap
            rsnew = np.dot(r.T,r)
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew/rsold)*p
            rsold = rsnew
        return x
    def LBFGS(self,alpha,nu,xi,i):
        x0 = self.x0.flatten()
        self.x0 = op.minimize(self.phi, x0, args=(alpha,nu,xi,i), method='L-BFGS-B', jac=self.gradPhi,options={'disp': None, 'gtol': int(np.sqrt(len(x0)))*1e-06/(1+np.exp(xi))**2},tol= 1e-8).x
        self.gtol = 0
        return np.squeeze(np.asarray(self.x0))

    def solver(self):
        epsilon = 1e-1
        rho = 0.9
        inner = []
        q = []
        p = []
        theta = [self.alpha,self.nu,self.xi]
        #self.max_iter
        for k in range(5):
            for i in range(np.shape(self.b)[2]):
                upperObjectiveTemp = []
                inner.append(self.FISTA(self.alpha,self.nu,self.xi,i,epsilon))
                upperObjectiveTemp.append(np.linalg.norm(inner[-1]- np.squeeze(np.asarray(self.b[:,:,i].flatten())))**2)
                print("FISTA")
                q.append(self.CG(self.alpha,self.nu,self.xi,i,self.x0[i,:],2*(inner[-1]- self.b[:,:,i].flatten()),tol=epsilon))
                print("CG")
                gradPhi12 = np.concatenate((np.reshape(self.gradTV2D(inner[-1],self.nu),(self.matsize[0]**2,1)),np.reshape(self.alpha*self.partialNuGradTV(inner[-1],self.nu),(self.matsize[0]**2,1)),np.reshape(inner[-1],(self.matsize[0]**2,1))),axis=1)
                print("Grad12")
                gradg2 =0
                p.append(np.dot(gradPhi12.T ,q[-1]))
            pk = (1/np.shape(self.b)[2])*sum(np.array(p))
            print("Pk",pk)
            self.lowerIter.append(self.lowerIterSum)
            self.upperObjective.append(np.sum(upperObjectiveTemp))
            L = np.linalg.norm(pk)
            print(theta)
            theta = theta - (1/L)* pk
            self.alpha = theta[0]
            self.nu = theta[1]
            self.xi = theta[2]
            print("theta "+str(k+1)+ ": ",theta)
            inner = []
            if np.linalg.norm(pk)**2 < epsilon:
                return theta
            if epsilon > 1e-12:
                epsilon = rho*epsilon
        return theta
    
    
#main
path = '/Users/sadegh/Downloads/Humans/256*256'

#np.random.seed(0)

data = np.load(path+ '/' +'Data.npz')
b = data['X']
b = b/255
b = b[:,:,0:10]

A = np.eye(b.shape[1])
NoisyData = np.load(path+ '/' +'DataNoisy2.npz')
y = NoisyData['y']
y = y/255
x0 = y[:,:,3]
#np.random.rand(np.shape(b)[0],np.shape(b)[1])
y = y[:,:,0:10]



Bilevel = HOAG(A, b, y, x0, 0.1, 1.2, -2,-5,-4)
result = Bilevel.solver()
thetaOptimal = result
print("Theta:" +str(thetaOptimal))


plt.plot(Bilevel.lowerIter,Bilevel.upperObjective,color='magenta')
plt.xlabel('Lower Level Iteration')
plt.ylabel('Upper Level Objective Value')
plt.xscale('log')
plt.yscale('log')