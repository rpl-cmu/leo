import gtsam
import numpy as np
from collections import defaultdict

class Logger():

    def __init__(self, params=None):
        self.params = params
        self.data = defaultdict(dict)
        self.runtime = None

    def data(self):
        return self.data

    def params(self):
        return self.params

    def runtime(self):
        return self.runtime
    
    def log_step(self, name, val, step):
        self.data[step][name] = val
    
    def log_param(self, name, val):
        self.params[name] = val
    
    def log_runtime(self, runtime):
        self.runtime = runtime