#!/usr/bin/python
"""
Simulates a quarterly review process where employees in the bottom X percent of a company
are fired and replaced for several quarters.

This tool might be used to determine the combination of quarterly review process attributes
that lead to increased company production.

In the simulation, employees in each quarter contribute a total amount of 'production' to
the company's total. Managers then estimate (with assessment error) how much each employee 
produced. The employees are sorted based on the manager's rating, and the bottom X percent 
of the employees are put in a 'misses' bucket.

If employees are grouped in the misses bucket for a specified number of quarters within
a larger window of quarters, the employee is fired.

When an employee is fired, that employee's slot remains empty for a specified number of
quarters. When the employee slot is filled with a new hire, he produces at a specified
lower level for a quarter. The simulation can also simulate drawing new hires from a
population with improved average productivity over the existing population (subject to
manager assessment error).

Output values are total average company production and a histogram of the production of
fired employees for each policy configuration. Presumably, one would select the policy
that maximized productivity while minimizing firing.

To run:

python2.7 simulate_qpr.py | python2.7 -mjson.tool

"""

import argparse, itertools, random, math, functools, multiprocessing, json
import numpy as np

def fire_employees(employees, firing_productivity_penalty, num_misses_allowed, quarter_window, quarter):
  """
  Fire employees who are eligible for firing
  Args:
    employees: population of employees
    firing_productivity_penalty: penalty to apply to the production of future quarters
    num_misses_allowed: num of times within the quarter window that an employee can be in the miss bucket before being fired
    quarter_window: number of quarters for which num misses is calculated (ex: 2 miss ratings allowed in 4 quarters)
    quarter: the current quarter
  Returns:
    fired_production - a list of individual employee quarterly production lost through firing
  """

  fired_production = []
  for e in employees:

    # can only fire active employee slots
    if( e['active'][quarter] == 'active'):
      quarter_miss_count = e["num_misses"]

      # employee slot is eligible for firing
      if( quarter_miss_count >= num_misses_allowed ):
        e['active'][quarter+1] = 'inactive'
        fired_production.append(e['actual_production'])

        # apply productivity penalty to account for time to recruit and replace production
        for j in range(0,firing_productivity_penalty):
          e['active'][quarter+1 + j] = 'inactive'
          e["num_misses"] = 0

        # schedule this employee slot to be rehired
        e['active'][quarter+1 + firing_productivity_penalty] = 'rehire'
      else:
        # rehire this employee slot next quarter
        if( (quarter+1) not in e['active'] ):
          e['active'][quarter+1] = 'active'

  return fired_production

def bucket_employees(employees, bin_size, miss_bucket_percentage, quarter, manager_can_veto):
  """
  Put employee slots into ratings buckets based on their production relative to peers
  Args:
    employees: employee slots from which production happens
    bin_size: size of company divisions where ratings distributions are enforced
    miss_bucket_percentage: percent of employees to give a fireable rating
    quarter: current quarter
    manager_can_veto: can a manager override an inaccurate rating in the miss bucket
  """

  # shortcut this function if we aren't firing anyone in this configuration,
  # which is the case when miss_bucket_percentage < 0
  if( miss_bucket_percentage < 0 ):
    for e in employees:
      e['buckets'][quarter] = 'achieves'
    return

  # for each division in the company
  bin_start = 0
  while( bin_start < len(employees) ):

    # get the employees in the division
    bin_employees = employees[bin_start:(bin_start+bin_size)]

    # debug
    #if( len(bin_employees) == 0 ):
    #  print bin_start,bin_size, len(employees)

    # sort employees in this division by their ratings (estimate of actual production)
    ratings = []
    for e in bin_employees:
      if( e['active'][quarter] == 'active'):
        ratings.append(e['ratings'][quarter])
    ratings.sort()

    # find the index where that marks the border between the firing bucket and the retaining bucket
    idx = min( max(0,int(math.floor(len(ratings) * miss_bucket_percentage))), len(ratings) )

    # rating at the edge of the miss bucket
    cutoff_rating = ratings[idx]

    # put employees into the misses bucket if their rating is below the cutoff
    for e in bin_employees:
      if( e['active'][quarter] == 'active' and e['ratings'][quarter] < cutoff_rating ):

        # If a (lower-level) manager can veto, that means he or she has some special insight
        # into the actual productivity of the employee and knows they are better than their rating
        if(  manager_can_veto==0 or ( manager_can_veto==1 and e['actual_production'] < cutoff_rating ) ):
          e['buckets'][quarter] = 'misses'
          e['num_misses'] += 1
        else:
          e['buckets'][quarter] = 'achieves'

      elif( e['active'][quarter] == 'inactive' ):
        e['buckets'][quarter] = 'na'
      else:
        e['buckets'][quarter] = 'achieves'

    # go to the employees in the next division
    bin_start += bin_size

def simulate_qpr(num_quarters, num_reps, config, num_hist_bins = 5):
  """
  Simulate one policy configuration
  Args:
    num_quarters: simulate the rating process for this number of quarters
    num_reps: repeat the experiment this number of times
    config: the configuration variables and values for this experiment
    num_hist_bins: the number of bins to use in the production results reporting
  Returns:
    histogram: a histogram of the production level of fired employees
    production: total actual production for this simulation
  """

  results = []

  # setup results histogram
  fired_production_histogram = [0]*num_hist_bins
  sample_pop , prod_avg, prod_range = config['employee_generators']()
  hist_bins = [ i*(prod_range/num_hist_bins) for i in range(0,num_hist_bins+1) ]

  # Monte Carlo runs
  for i in range(0,num_reps):

    # generate employees using employee generator
    employees, prod_avg, prod_range = config['employee_generators']()

    result = {'employee_generators':employees}

    # total actual production over simulation period
    production = 0
    for j in range(0,num_quarters):

      ## Add production of each employee
      for e in employees:

        # Replace employees who are ready to be replaced
        if( e['active'][j] == 'rehire' ):
          e['active'][j] = 'active'

          # new hire productivity subject to manager's assessment ability
          e['actual_production'] = (prod_avg * (1.0+config['hiring_improvement_factor']) ) \
            + random.gauss(0, config['manager_review_error'] * prod_range/3.0 )

        # Produce
        if( e['active'][j] == 'active' ):
          production += e['actual_production']

          ## Rate quarter
          # rating is centered around actual production with error
          e['ratings'][j] = e['actual_production'] +  random.gauss(0, config['manager_review_error'] )

      # put employees into rating buckets
      bucket_employees(employees=employees,
        bin_size=config['bin_size'],
        miss_bucket_percentage=config['miss_bucket_percentage'],
        quarter=j,
        manager_can_veto=config['manager_veto'])

      # fire employees who meet the firing criteria
      fired_production = fire_employees(employees=employees,
        firing_productivity_penalty=config['firing_production_penalty'],
        num_misses_allowed=config['num_misses_allowed'],
        quarter_window=config['num_evaluation_quarters'],
        quarter=j)

      # create a histogram of the production level of fired employees
      y,x = np.histogram(fired_production,bins=hist_bins)
      for k in range(0,len(y)):
        fired_production_histogram[k] += y[k]

    result['production'] = production

    results.append(result)

  return results, fired_production_histogram

def create_employees(num_employees):
  """
  Helper function to create a list of employees slots.
  Employee slots can be active or inactive. Production from active employee slots
  contributes to the company's total production.
  Not using a class because employees are more like employee slots where a new
  new employee can replace an old one
  """

  employees = []
  for i in range(0,num_employees):
    employee = {}
    employee["ratings"] = {}
    employee["buckets"] = {}
    employee["num_misses"] = 0
    employee["active"] = {0:'active',}
    employees.append(employee)
  return employees

def generate_normal_distribution_of_employees(n,a,b):
  """
  normally distributes employee productivity
  returns the list of employees, their average production, and their range of production
  """

  num_employees = n
  avg = a
  three_std_dev = b

  employees = create_employees(num_employees)

  total_production = 0.0

  min_prod = 0.0
  max_prod = avg + three_std_dev

  for e in employees:
    e["actual_production"] = min(max(min_prod,random.gauss(avg, three_std_dev/3.0 )), max_prod)
    total_production += e["actual_production"]

  return employees, total_production/float(len(employees)), max_prod-min_prod

def generate_linear_distribution_of_employees(n, a,b):
  """
  distributes employee productivity uniformly between a and b
  returns the list of employees, their average production, and their range of production
  """
  num_employees = n
  low = a
  high = b

  inc = (high-low)/(num_employees)

  employees = create_employees(num_employees)

  total_production = 0.0
  running_level = 0
  for e in employees:
    running_level += inc
    total_production += running_level
    e["actual_production"] = running_level + low

  random.shuffle(employees)

  return employees, total_production/float(len(employees)), high-low

def execute_config(config,config_keys,num_reps):
  """
  Monte Carlo simulation of a single experiment configuration
  Args:
    config = a list of possible experiment variables
    config_keys = a list of experiment variables to include in the experiment
    num_reps = number of times to repeat this configuration
  """

  # make a dictionary out of the two lists for easy reference
  c = dict(zip(config_keys,config))

  # run the simulations
  results, fired_production_histogram = simulate_qpr(
    num_quarters=args.num_quarters,
    num_reps=args.num_reps,
    config=c )

  # tabulate the production results
  total_production = [r['production'] for r in results]

  # log results
  out = {'output':{},'input':{}}
  for k in c.keys():
    out['input'][k] = c[k]
  out['input']['employee_generators'] = "\"%s\"" % ( config[1].get_config_name() )
  out['output']['production'] = "%f" % (sum( total_production )/float(num_reps) )

  # tabulate the histogram of production level of employees of fired employees
  for i in range(0,len(fired_production_histogram)):
    out['output']["fired_production_bin_%d" % (i+1)] = str( fired_production_histogram[i] )

  return out

class EmployeeGenerator:
  """
  Helper class that generates a population of employees with randomly distributed productivity
  """

  def __init__(self, f, fargs ):
    self.function = f
    self.fargs = fargs

  def __call__(self):
    return self.function(**self.fargs)

  # used for reporting
  def get_config_name(self):
    return "%s(%s)" % (self.function.short_name,",".join( map(str,self.fargs.values()) ))


def run_experiment(setup,config_keys,num_reps,pool_size=(multiprocessing.cpu_count()-1)):
    """
    Top-level function for executing all the configurations in an experiment
    """

    # do cartesian product of all experiment setup variables
    configs = list( itertools.product(*setup) )

    # create a unique id for each configuration    
    id = 0
    new_configs = []
    for config in configs:
      c = [id]
      c.extend(config)
      new_configs.append(tuple(c))
      id += 1

    # simulate each configuration
    pool = multiprocessing.Pool(pool_size)
    data = pool.map(functools.partial(execute_config,num_reps=num_reps,config_keys=config_keys),
      new_configs)

    # non parallel way
    #data = []
    #for config in configs:
    #  data.append(execute_config(config_keys,config, num_reps))

    return data

if __name__ == "__main__":

  # used for printing the experiment setup in the results
  generate_linear_distribution_of_employees.short_name = "lin"
  generate_normal_distribution_of_employees.short_name = "norm"

  parser = argparse.ArgumentParser(description='Simulate a quarterly review process where employees in the bottom X percent of a company for several quarters are fired and replaced')
  parser.add_argument('--num-quarters', default=12, required=False, type=int, help='Number of quarters to simulate')
  parser.add_argument('--num-reps', default=10, required=False, type=int, help='Number of repetitions for simulation')
  parser.add_argument('--num-cpus', default=(multiprocessing.cpu_count()-1), required=False, type=int, help='Number of cpus to use')
  args = parser.parse_args()

  ## Setup parameterized study

  # company size
  num_employees = 5000

  # employee talent distribution - normal and linear distributions
  # for example in a normal distribution:
  # a = average production per quarter
  # b = 3x std deviation in production per quarter
  employee_generators = [
    #EmployeeGenerator(generate_linear_distribution_of_employees, {"n":num_employees,"a":0.0,"b":5.0} ),
    #EmployeeGenerator(generate_linear_distribution_of_employees, {"n":num_employees,"a":0.0,"b":2.5} ),
    #EmployeeGenerator(generate_linear_distribution_of_employees, {"n":num_employees,"a":0.0,"b":1.0} ),
    #EmployeeGenerator(generate_normal_distribution_of_employees, {"n":num_employees,"a":2.5,"b":1.25} ),
    #EmployeeGenerator(generate_normal_distribution_of_employees, {"n":num_employees,"a":2.5,"b":2.5} ),
    #EmployeeGenerator(generate_normal_distribution_of_employees, {"n":num_employees,"a":5.0,"b":5.0} ),
    #EmployeeGenerator(generate_normal_distribution_of_employees, {"n":num_employees,"a":2.5,"b":2.5} ),
    #EmployeeGenerator(generate_normal_distribution_of_employees, {"n":num_employees,"a":10.0,"b":10.0} ),
    #EmployeeGenerator(generate_normal_distribution_of_employees, {"n":num_employees,"a":25.0,"b":25.0} ),
    EmployeeGenerator(generate_normal_distribution_of_employees, {"n":num_employees,"a":50.0,"b":50.0} ),
  ]

  # size of groups of employees by which miss percentages are applied
  # meant to simulate different divisions rating and firing their employees
  # independently
  bin_sizes = [ num_employees ,num_employees/10,num_employees/100,num_employees/500 ]

  # manager rating error
  manager_review_stdev = [0,.1,.25]

  # number of quarters of lost production after firing
  firing_production_penalty = [0,2,4]

  # percentage of employees that will be placed in the lowest buckets
  miss_bucket_percentage = [.15,0.05]

  # number of ratings in the miss buckets during an evaluation period that makes an employee
  # eligible for firing
  num_misses_allowed = [2,3]

  # sliding window of quarters by which to evaluate an employee for firing
  num_evaluation_quarters = [5]

  # the average increase in new hire production over the existing population's production average
  hiring_improvement_factor = [0,0.1,0.25]

  # if 1 and if an employee's actual production is above the rating, the employee's rating
  # is revised up to match the true value. This is meant to simulate the case where a
  # lower level manager with better information about an employee fights to increase
  # a rating out of the firing bucket
  manager_veto = [0,1]

  config_keys = ['id',
    'employee_generators',
    'bin_size',
    'manager_review_error',
    'miss_bucket_percentage',
    'firing_production_penalty',
    'num_misses_allowed',
    'num_evaluation_quarters',
    'hiring_improvement_factor',
    'manager_veto',
  ]

  # Run simulation
  #print "\t".join(config_keys),"\t","avg_production"


  # control case - no firing
  control_setup = [employee_generators,
    [num_employees], # bin sizes
    [0], # manager review stdev
    [-1], # miss bucket percentage
    [0], # firing_production_penalty
    [5], # num_misses_allowed
    [5], # num_evaluation_quarters
    [0], # hiring_improvement_factor
    [0] #manager_veto
  ]

  results = {}

  results['without_firing'] = run_experiment(setup=control_setup,
    config_keys=config_keys,
    num_reps=args.num_reps,
    pool_size=args.num_cpus
    )


  # firing cases - small configuration
  setup = [employee_generators,
    [num_employees], # bin sizes
    [.1], # manager review stdev
    [0.05], # miss bucket percentage
    [2], # firing_production_penalty
    [2], # num_misses_allowed
    [5], # num_evaluation_quarters
    [0, .1], # hiring_improvement_factor
    [0] #manager_veto
  ]

#  # firing cases - large configuration
#  setup = [employee_generators,
#    bin_sizes ,
#    manager_review_stdev,
#    miss_bucket_percentage,
#    firing_production_penalty,
#    num_misses_allowed,
#    num_evaluation_quarters,
#    hiring_improvement_factor,
#    manager_veto
#  ]

  results['with_firing'] = run_experiment(setup=setup,
    config_keys=config_keys,
    num_reps=args.num_reps,
    pool_size=args.num_cpus
  )

  print json.dumps(results)
