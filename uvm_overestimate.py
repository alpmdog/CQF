from uvm import *

model = modelparams(0.02, 100, 0.1, 0.3)
constant_low = modelparams(0.02, 100, 0.1, 0.1)
constant_high = modelparams(0.02, 100, 0.3, 0.3)
digital_call = security('digital call', 110, 1, 1)

p = Portfolio()
p.add(digital_call)

fdmSolver = ExplicitFDMSolver(100, model)
fdmSolver.solve(p)
results = fdmSolver.get_results()
results.to_csv("dcallworst.csv")

fdmSolver = ExplicitFDMSolver(100, constant_low)
fdmSolver.solve(p)
results = fdmSolver.get_results()
results.to_csv("dcalllow.csv")

fdmSolver = ExplicitFDMSolver(100, constant_high)
fdmSolver.solve(p)
results = fdmSolver.get_results()
results.to_csv("dcallhigh.csv")

fdmSolver = ExplicitFDMSolver(100, model, worst=False)
fdmSolver.solve(p)
results = fdmSolver.get_results()
results.to_csv("dcallbest.csv")

