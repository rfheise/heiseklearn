from ..GLM.Logistic import Logistic
import numpy as np
from scipy.stats import multivariate_normal as norm
from ...Tests import BankTest,TitanicTest, Test 
import multiprocessing as mp
#metropolic hastings algo to estimate posterior for logisitc regression
class MH(Logistic):

    def __init__(self, burn_in=100, samples=100, chains=5, max_proc=3):
        self.burn_in = burn_in
        self.samples = samples 
        self.chains = chains
        self.max_proc = max_proc

    def train(self,X,y):
        # self.train_single_proc(X,y)
        self.train_multi_proc(X,y)

    def train_single_proc(self, X, y):
        theta = 0
        for i in range(self.chains):
            print(f"Running chain {i}")
            theta += self.one_mcmc(X,y)

        self.theta = theta/self.chains

    def train_multi_proc(self,X,y):
        
        theta = 0
        queue = []
        for i in range(self.chains):
            print(f"Running chain {i}")
            parent_conn,child_conn  = mp.Pipe()
            proc = mp.Process(target=self.one_mcmc_proc, args=(X,y,child_conn))
            queue.append((proc, parent_conn))
            proc.start()
            if len(queue) >= self.max_proc:
                chain = queue.pop()
                theta += chain[1].recv()
                chain[0].join()
        while len(queue) != 0:
            chain = queue.pop()
            theta += chain[1].recv()
            chain[0].join()
                
        self.theta = theta/self.chains
    
    def one_mcmc_proc(self, X, y, conn):
        val = self.one_mcmc(X,y)
        conn.send(val)
        conn.close()

    def one_mcmc(self, X, y):
        self.theta = np.random.normal(loc=0,scale=.1,size=(X.shape[1],1))
        
        #burn in 
        print("Starting Burn in")
        for i in range(self.burn_in):
            while True:
                theta_prime = self.sample(X,y)
                if theta_prime is not None:
                    self.theta = theta_prime
                    break
        
        samples = 0
        print("Collecting Samples")
        for i in range(self.samples):
            while True:
                theta_prime = self.sample(X,y)
                if theta_prime is not None:
                    self.theta = theta_prime 
                    break
            samples += self.theta
                
        return samples/self.samples

    @staticmethod
    def pdf(x, mean, cov):
        exp = -.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
        const = np.log(1/((2 * np.pi)**(cov.shape[0]/2) * np.linalg.norm(cov)))
        return exp + const

    def sample(self, X, y):
        X = X.to_numpy()
        y = np.reshape(y, (y.shape[0],1))
        dims = self.theta.shape[0]
        d = np.zeros((self.theta.shape[0],1))
        #transition prob as gaussian with mu theta_(t-1)
        theta_prime = np.random.normal(loc=self.theta,scale=.1)
        # assume prior is a gaussian centered around 0
        prior_theta_prime = self.pdf(theta_prime, mean=d, cov= np.eye(dims))
        hypo_prime = self.hypothesis(X @ theta_prime)
        ll_prime = y.T @ np.log(hypo_prime) + (1 - y.T) @ np.log(1 - hypo_prime)
        trans_theta = self.pdf(self.theta, mean=theta_prime, cov = np.eye(dims))
        log_num = ll_prime + prior_theta_prime + trans_theta

        prior_theta = self.pdf(self.theta, mean=d, cov=np.eye(dims))
        hypo_theta = self.hypothesis(X @ self.theta)
        ll_theta = y.T @ np.log(hypo_theta) + (1 - y.T) @ np.log(1 - hypo_theta)
        trans_theta = self.pdf(theta_prime, mean=self.theta, cov = np.eye(dims))
        log_den = prior_theta + ll_theta + trans_theta 
        prob = min(1, np.exp(log_num - log_den))
        if np.random.uniform() > (1 - prob):
            return theta_prime 
        return None
    
if __name__ == "__main__":

    model = MH()
    test = TitanicTest(model)
    test.run_benchmarks()

