#load benchmarks problems

import json
import numpy as np
import ray
from gradient_free_optimizers import RandomSearchOptimizer,RandomRestartHillClimbingOptimizer,RandomAnnealingOptimizer,PatternSearch,PowellsMethod,GridSearchOptimizer
from gradient_free_optimizers import HillClimbingOptimizer,StochasticHillClimbingOptimizer,RepulsingHillClimbingOptimizer,SimulatedAnnealingOptimizer,DownhillSimplexOptimizer
from gradient_free_optimizers import ParallelTemperingOptimizer,ParticleSwarmOptimizer,EvolutionStrategyOptimizer,SpiralOptimization
from gradient_free_optimizers import ForestOptimizer,BayesianOptimizer,TreeStructuredParzenEstimators,LipschitzOptimizer,DirectAlgorithm


class prb:   #this will be your black box function
    def __init__(self,prob,noisy_level):
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
            #a mathematical function or a python function which will be called.
            y = eval(self.func)
        except:
            x = np.array([x])
            y = eval(self.func)
        return float(y)
    def constraint(self,x):
        # Appears to evaluate if the constraints are satisfied
        #returns 1 if all constraints are satisfied, else -1
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
    
bcp=np.load('./group_src/bcp8.npy',allow_pickle=True).item()

class benchmark:
    def __init__(self,func,noisy_level):
        self.func = str(func)
        self.nl = noisy_level
    def datcol(self,x):
        try:
            y = eval(self.func)
        except:
            x = np.array([x])
            y = eval(self.func)
        return float(y)

############ ----------------- Lucas Helper Functions ------------------ #############

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
        #clear_output(wait=True)
        x = search_space
        return -1*eval(opt_form)
    return objective_fxn


############ ----------------- My Helper Functions ------------------ #############


def get_valid_problems(names):
    """ 
    Some provided solutions may be invalid. This function returns valid/invalid solutions
    """

    valid_problems = []
    invalid_problems = []
    for name in names: 
        dims = bcp[f'{name}']['n']
        try:
            assert len(bcp[f'{name}']['xub']) == dims and len(bcp[f'{name}']['xlb']) == dims
            assert len(bcp[f'{name}']['xopt']) == dims 
            valid_problems.append(name)
        except:
            invalid_problems.append(name)
            continue

    print(f"There are {len(valid_problems)} valid problems and {len(invalid_problems)} invalid problems")
    
    return valid_problems, invalid_problems

def get_perturbed_solutions(test_fxn):

    """
    Returns perturbed optimal solutions for a given input problem 
    """

    dims = test_fxn['n']
    perturbed_solutions = []

    for dim in range(dims):
        opt = test_fxn['xopt'][dim]       
        perturbation = np.random.uniform(-5, 5) / 100
        if opt != 0.0: perturbed_solution = np.array([opt + opt*perturbation])
        else: perturbed_solution = np.array([perturbation])
        perturbed_solutions.append(perturbed_solution)

    return np.concatenate(perturbed_solutions)

def get_all_initial_guesses(names):

    """
    Returns perturbed initial guesses for all valid problems as a list of dicts
    """

    valid_problems, invalid_problems = get_valid_problems(names)
    all_initial_guesses = []

    for problem in valid_problems:
        test_fxn = bcp[f'{problem}']
        dims = test_fxn['n']
        perturbed_solution = get_perturbed_solutions(test_fxn).astype(float)
        optimal_solution = test_fxn['xopt'].astype(float)
        percent_diff = np.divide(abs(perturbed_solution - optimal_solution), abs(optimal_solution), 
                                out=np.zeros_like(optimal_solution), where=optimal_solution!=0) * 100.0
        assert max(percent_diff) <= 5 # we sampled perturbations from uniform(-5,5)
        assert len(perturbed_solution) == dims

        initial_guess = {str(i):perturbed_solution[i] for i in range(dims)}
        all_initial_guesses.append(initial_guess)

    return all_initial_guesses, valid_problems, invalid_problems


def get_solutions(problems, solvers, all_initial_guesses):

    names = list(bcp.keys())
    final_solns = {}

    
    for problem in problems:

        try:
            test_fxn = bcp[f'{problem}']
            fxn_form = test_fxn['func']
            xlb = test_fxn['xlb']
            xub = test_fxn['xub']
            ndim = test_fxn['n']

            opt_form = get_usable_fxn(ndim, fxn_form)
            search_space = create_search_space(ndim, xlb, xub, resolution=100)
            fxn = create_parent_fxn(search_space,opt_form)

            initial_guess = all_initial_guesses[names.index(f'{problem}')]
            initialize={"warm_start":[initial_guess]}

            optimum = {}
            for f in solvers:
                opt = f(search_space) # initialize optimizer
                opt.search(fxn, n_iter=2500) # run optimizer
                optimum[f.__name__] = opt.best_para

            opts = ['Solver\tOptimum']
            for i in optimum.keys():
                tmp_opt = optimum[i]
                opts += ['{}\t{}'.format(i,tmp_opt)]
            with open(f'./{problem}-refinement.txt', 'w') as f:
                print('\n'.join(opts),file=f)

            final_solns[f'{problem}'] = optimum

        except: # some additional unexpected failures (e.g., OverFlow errors)
            problems = list(filter(lambda s: not (s[:]==f"{problem}"), problems))
            continue

    successful_problems = problems

    return final_solns, successful_problems


def get_solver_names(solvers):

    """ 
    Function to process strings of solvers
    """

    solver_names = []
    for i in range(len(solvers)):
        solver_name = str(solvers[i])
        # All solver names are after 4th "." 
        idx = solver_name.find(".", solver_name.find(".", solver_name.find(".", solver_name.find(".") + 1) + 1) + 1)

        # Select the substring after the 4th dot
        solver_name = solver_name[idx+1:]
        solver_name = solver_name.replace("'", "")
        solver_name = solver_name.replace(">", "")
        solver_names.append(solver_name)
    
    return solver_names

def get_maes(problems, solver_names, solver_solutions):

    """
    Get MAEs for a list of specified problems, ground truth solutions, and solution obtained
    by a list of solvers
    """
    from sklearn.metrics import mean_absolute_error

    maes = {}

    for solver_name in solver_names:
        solver_opts = [v[f'{solver_name}'] for v in solver_solutions.values()]
        true_opts = [bcp[f'{problem}']['yopt'] for problem in problems]
        mae = mean_absolute_error(solver_opts, true_opts)
        maes[f'{solver_name}'] = mae

    return maes

def create_function(problem_idx, func_string, valid_problems, invalid_problems):
    """
    Weird function to obtain an actual python function we can evaluate given inputs.
    I wrote this because I was unaware of the built in package evaluation
    """
    filtered_bcp = {key: value for key, value in bcp.items() if not any(s in key for s in invalid_problems)}

    for i in range(len(filtered_bcp[valid_problems[problem_idx]]['xopt'])):
        func_string = func_string.replace(f"x[:,{i}]",f"x[{str(i)}]")

    function_def = "def problem_function(x):\n    return " + func_string
    exec(function_def, globals())

    return globals()['problem_function']

def get_x_values(initial_guesses):
    """ 
    Convert initial guesses from dict to list for input to function
    """
    x = []
    for value in initial_guesses.values():
        x.append(value)
    return x

def evaluate_refinement(problems, solver_names, all_initial_guesses, solver_solutions, valid_problems, invalid_problems, taus):

    """
    Evaluate the refinement according to the equation from the paper
    """

    filtered_bcp = {key: value for key, value in bcp.items() if not any(s in key for s in invalid_problems)}

    for problem in problems:
        for solver in solver_names:          
            func_string = filtered_bcp[problem]['func'] 
            problem_idx = list(filtered_bcp.keys()).index(problem)
            initial_guesses = all_initial_guesses[problem_idx]
            initial_guesses  = get_x_values(initial_guesses)
            problem_function = create_function(problem_idx, func_string, valid_problems, invalid_problems)
            f_x0 = problem_function(initial_guesses)
            solver_output = solver_solutions[problem][solver]
            solver_output  = get_x_values(solver_output)
            f_solver = problem_function(solver_output)
            f_L = filtered_bcp[problem]['yopt']

            success = []

            for j in range(len(taus)):
                try:
                    assert f_x0 - f_solver >= (1 - taus[j])*(f_x0 - f_L)
                    success.append("successful")
                except:
                    success.append("not successful")
                    pass

            solver_solutions[problem][solver] = success
    
    return solver_solutions

def get_plot_dict(problems, solver_solutions, solver_names, taus):

    """
    Get results in format for plotting
    """

    # New dictionary more intuitive in ordering 
    # Note to self: This is ugly ugly ugly code
    solver_solutions_inverted = {}

    for outer_key, inner_dict in solver_solutions.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in solver_solutions_inverted:
                solver_solutions_inverted[inner_key] = {}
            solver_solutions_inverted[inner_key][outer_key] = value

    solver_solutions = solver_solutions_inverted 
    plot_dict = {}

    for solver in solver_names:
        num_successfuls = [0] * len(taus)
        for inner_list in solver_solutions[solver].values():
            for i, value in enumerate(inner_list):
                if value == 'successful':
                    num_successfuls[i] += 1

        fraction = [x / len(problems) for x in num_successfuls]
        plot_dict[solver] = fraction

    return plot_dict

@ray.remote
def run_refinement(problems, solvers, all_initial_guesses, valid_problems, invalid_problems, taus):

    solver_names = get_solver_names(solvers)
    solver_solutions, successful_problems = get_solutions(problems, solvers, all_initial_guesses)
    solver_refinement = evaluate_refinement(successful_problems, solver_names, all_initial_guesses, solver_solutions, valid_problems, invalid_problems, taus)
    plot_dict = get_plot_dict(successful_problems, solver_refinement, solver_names, taus)

    with open('./successful-problems.txt','w') as tfile:
        tfile.write('\n'.join(successful_problems))

    return plot_dict

def run_refinement_local(problems, solvers, all_initial_guesses, valid_problems, invalid_problems, taus):

    solver_names = get_solver_names(solvers)
    solver_solutions, successful_problems = get_solutions(problems, solvers, all_initial_guesses)
    solver_refinement = evaluate_refinement(successful_problems, solver_names, all_initial_guesses, solver_solutions, valid_problems, invalid_problems, taus)
    plot_dict = get_plot_dict(successful_problems, solver_refinement, solver_names, taus)

    with open('./successful-problems.txt','w') as tfile:
        tfile.write('\n'.join(successful_problems))

    return plot_dict


if __name__ == "__main__":

    global_opt = [RandomSearchOptimizer,RandomRestartHillClimbingOptimizer,RandomAnnealingOptimizer,PatternSearch,PowellsMethod,GridSearchOptimizer]
    local_opt = [HillClimbingOptimizer,StochasticHillClimbingOptimizer,RepulsingHillClimbingOptimizer,SimulatedAnnealingOptimizer,DownhillSimplexOptimizer]
    pop_opt = [ParallelTemperingOptimizer,ParticleSwarmOptimizer,EvolutionStrategyOptimizer,SpiralOptimization]

    names = list(bcp.keys())
    all_initial_guesses, valid_problems, invalid_problems = get_all_initial_guesses(names)
    problems = valid_problems
    solvers = [global_opt,local_opt,pop_opt]

    taus = [1e-3, 1e-4, 1e-5, 1e-7, 0e+0]

    try:
        ray.init()
    except RuntimeError: # cluster is already initialized
        pass
    
    futures = [run_refinement.remote(problems, solvers[i], all_initial_guesses, valid_problems, invalid_problems, taus) for i in range(len(solvers))]
    plot_dicts = ray.get(futures)

    with open('./dicts-all.txt', 'w') as convert_file:
        convert_file.write(json.dumps(plot_dicts))