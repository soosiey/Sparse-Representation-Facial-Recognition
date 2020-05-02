import cvxpy as cp

class recognizer():

    def __init__(self, dataset):
        self.data = dataset
        self.train = dataset.train[:,1:].T


    def solve(self, b, y):


        self.x = cp.Variable(b.shape[0])
        self.obj = cp.Minimize(cp.norm(self.x,1))
        self.constraints = [self.train*self.x == y]
        self.prob = cp.Problem(self.obj, self.constraints)
        self.prob.solve()
    def getOptim(self):
        return self.x.value
    def getOptimVal(self):
        return self.prob.value
    def getStatus(self):
        return self.prob.status

