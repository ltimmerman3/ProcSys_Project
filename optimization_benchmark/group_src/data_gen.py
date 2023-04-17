# load benchmarks problems

import numpy as np

bcp = np.load('bcp8.npy', allow_pickle=True).item()
eqc = np.load('eqcS.npy', allow_pickle=True).item()

class prb:   # this will be your black box function
    def __init__(self, prob, noisy_level):
        # function for generating data
        self.func=prob['func']
        # number of constraints
        self.n_con=prob['n_con']
        # noise level
        self.nl=noisy_level
        # constraints
        self.con=[]
        for i in range(self.n_con):
            self.con.append(prob['con'+str(i+1)])
    def datcol(self,x):
        try:
            # eval takes string type input and evaluates python code. Input can be
            # a mathematical function or a python function which will be called.
            y = eval(self.func)
        except:
            x = np.array([x])
            y = eval(self.func)
        return float(y)
    def constraint(self, x):
        # Appears to evaluate if the constraints are satisfied
        # returns 1 if all constraints are satisfied, else -1
        try:     
            m,n=np.shape(x)
        except:
            x = np.array([x])
            m,n=np.shape(x)
        y=np.empty((self.n_con))
        for i in range(self.n_con):
            y[i]=eval(self.con[i])     
        if y.all() == 1.:
            label = 1.
        else:
            label = -1.
        return label

class benchmark:
    def __init__(self, func, noisy_level):
        self.func = str(func)
        self.nl = noisy_level
    def datcol(self, x):
        try:
            y = eval(self.func)
        except:
            x = np.array([x])
            y = eval(self.func)
        return float(y)
    
import numbers

def get_valid_problems(source, names):
    """ 
    Some provided solutions may be invalid. This function returns valid/invalid solutions
    """

    valid_problems = []
    invalid_problems = []
    for name in names: 
        dims = source[f'{name}']['n']
        try:
            assert len(source[f'{name}']['xub']) == dims and len(source[f'{name}']['xlb']) == dims
            assert len(source[f'{name}']['xopt']) == dims and isinstance(source[f'{name}']['yopt'], numbers.Number)
            valid_problems.append(name)
        except:
            invalid_problems.append(name)
            continue
    
    return valid_problems, invalid_problems

names_bcp = list(bcp.keys())
names_eqc = list(eqc.keys())
bcp_valid_probs, bcp_invalid_probs = get_valid_problems(bcp, names_bcp)
eqc_valid_probs, eqc_invalid_probs = get_valid_problems(eqc, names_eqc)

all_valid_probs = bcp_valid_probs #+ eqc_valid_probs
all_valid_probs.sort()

# Import all optimizers explicitly into namespace
# exp_opt
from gradient_free_optimizers import EnsembleOptimizer
exp_opt = [EnsembleOptimizer]
#global_opt
from gradient_free_optimizers import RandomSearchOptimizer,RandomRestartHillClimbingOptimizer,RandomAnnealingOptimizer,PatternSearch,PowellsMethod,GridSearchOptimizer
global_opt = [RandomSearchOptimizer,RandomRestartHillClimbingOptimizer,RandomAnnealingOptimizer,PatternSearch,PowellsMethod,GridSearchOptimizer]
#local_opt
from gradient_free_optimizers import HillClimbingOptimizer,StochasticHillClimbingOptimizer,RepulsingHillClimbingOptimizer,SimulatedAnnealingOptimizer,DownhillSimplexOptimizer
local_opt = [HillClimbingOptimizer,StochasticHillClimbingOptimizer,RepulsingHillClimbingOptimizer,SimulatedAnnealingOptimizer,DownhillSimplexOptimizer]
#pop_opt
from gradient_free_optimizers import ParallelTemperingOptimizer,ParticleSwarmOptimizer,EvolutionStrategyOptimizer,SpiralOptimization
pop_opt = [ParallelTemperingOptimizer,ParticleSwarmOptimizer,EvolutionStrategyOptimizer,SpiralOptimization]
#smb_opt
from gradient_free_optimizers import ForestOptimizer,BayesianOptimizer,TreeStructuredParzenEstimators,LipschitzOptimizer,DirectAlgorithm
smb_opt = [ForestOptimizer,BayesianOptimizer,TreeStructuredParzenEstimators,LipschitzOptimizer,DirectAlgorithm]

all_opt = {'exp_opt':exp_opt,'global_opt':global_opt,'local_opt':local_opt,'pop_opt':pop_opt,'smb_opt':smb_opt}

# Find and replace function for editing function strings
def multiple_replace(_string:str, find_replace:list):
    new_string = _string
    for find,replace in find_replace:
        new_string = new_string.replace(find,replace)
    return new_string

# Create usable form of the function to pass to optimizers
def get_usable_fxn(dimension: int, fxn_form: str):
    find_rep_list = [('x[:,{}]'.format(dimension-1-i),'x["{}"]'.format(dimension-1-i)) for i in range(dimension)]
    opt_form = multiple_replace(fxn_form, find_rep_list)
    return opt_form

# Create search space
def create_search_space(dimension: int, xlb: list, xub: list, resolution: int = 100):
    search_space = {str(i):np.linspace(xlb[i],xub[i],resolution) for i in range(dimension)}
    return search_space

# Create a container for objective function - need to use eval to evaluate string version of function, but can't pass that
#directly to optimizer - there is probably a better way to do this, possibly using wrappers. Not worth time right now.
def create_parent_fxn(search_space,opt_form):
    search_space = search_space
    opt_form = opt_form
    def objective_fxn(search_space):
        x = search_space
        return -1*eval(opt_form)
    return objective_fxn

import tqdm
import sys

opt_type = sys.argv[1]
do_opt = all_opt[opt_type]
#taus = [1e-1,1e-2,1e-3,1e-6,0]
tmp_list = [f"{opt.__name__}\t" for opt in do_opt]+["f_x0"]
result_list = []
result_list += [''.join(tmp_list)]

for _prob in tqdm.tqdm(all_valid_probs, desc="Problems"):
    test_fxn = bcp[_prob]
    fxn_form = test_fxn['func']
    xlb = test_fxn['xlb']
    xub = test_fxn['xub']
    ndim = test_fxn['n']
    y_opt = test_fxn['yopt']
    x_opt = test_fxn['xopt']
    scales = [xub[i]-xlb[i] for i in range(ndim)]
    try:
        initial_guess = {str(i):np.random.normal(loc=x_opt[i],scale=scales[i]/3) for i in range(ndim)}
    except:
        tmp_list = []
        for i in range(len(do_opt)):
            tmp_list += [f"{np.nan}\t"]
        tmp_list += [f"{np.nan}"]
        result_list += [''.join(tmp_list)]
        continue
    x = initial_guess
    opt_form = get_usable_fxn(ndim, fxn_form)
    f_x0 = eval(opt_form)
    search_space = create_search_space(ndim, xlb, xub)
    parent_fxn = create_parent_fxn(search_space,opt_form)  
    medians = []
    for _opt in tqdm.tqdm(do_opt, desc="Optimizers"):
        f_solve = []
        #for i in range(10):
        opt = _opt(search_space, initialize={"warm_start":[initial_guess]})
        with open('initial_guesses','a') as tmp_f:
            print(initial_guess,file=tmp_f)
        try:
            opt.search(parent_fxn, n_iter=2500, verbosity=[False,False,False])
            f_solve.append(opt.best_score)
            f_solve.sort()
            median = np.median(f_solve)
            median = -1*median
            medians.append(median)
        except:
            medians.append(np.nan)
    tmp_list = [f"{_med}\t" for _med in medians]+[f"{f_x0}"]
    result_list += [''.join(tmp_list)]
with open(f"GivenSP_{opt_type}_Results.txt",'w') as f:
    f.write('\n'.join(result_list))