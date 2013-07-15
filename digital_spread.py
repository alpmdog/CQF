from uvm import *

model = modelparams(0.02, 100, 0.1, 0.3)
digital_call = security('digital call', 110, 1, 1)
call_100 = security('call', 100, 1, -0.05) #-0.05
call_120 = security('call', 120, 1, 0.05)

p = Portfolio()
p.add(digital_call)
p.add(call_100)
p.add(call_120)

fdmSolver = ExplicitFDMSolver(100, model)
fdmSolver.solve(p)
results = fdmSolver.get_results()

bsPricer = BSPricer(model)
hedging_cost = bsPricer.hedging_cost(p, 0.1)

print "hedging cost is %s" % hedging_cost
print get_value_bylevel(results, 100.) - hedging_cost

results.to_csv("dcall3_110.csv")

digital_call = security('digital call', 110, 1, -1)
call_100 = security('call', 100, 1, 0.05)
call_120 = security('call', 120, 1, -0.05)

p = Portfolio()
p.add(digital_call)
p.add(call_100)
p.add(call_120)

fdmSolver = ExplicitFDMSolver(100, model)
fdmSolver.solve(p)
results = fdmSolver.get_results()

hedging_cost = bsPricer.hedging_cost(p, 0.1)

print hedging_cost - get_value_bylevel(results, 100.)
