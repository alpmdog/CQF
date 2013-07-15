from uvm import *

model = modelparams(0.02, 100, 0.1, 0.3)
digital_call = security('digital call', 110, 1, 1)
digital_call_130 = security('digital call', 130, 1, 1)
digital_call_90 = security('digital call', 90, 1, 1)

p = Portfolio()
p.add(digital_call)
p.add(digital_call_130)
p.add(digital_call_90)

fdmSolver = ExplicitFDMSolver(100, model)
fdmSolver.solve(p)
results = fdmSolver.get_results()
results.to_csv("dcall_portfolio.csv")

#get_value_bylevel(results,100.)

fdmSolver = ExplicitFDMSolver(100, model)
fdmSolver.solve(p, solve_separately=True)
results = fdmSolver.get_results()
results.to_csv("dcall_separate.csv")

#get_value_bylevel(results,100.)
