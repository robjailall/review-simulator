Review Simulator

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

python simulate_qpr.py | python -mjson.tool
