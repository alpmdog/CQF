from uvm import *

model = modelparams(0.02, 100, 0.1, 0.3)
digital = security('call', 110, 1, 1)
call_100 = security('call', 100, 1, -1)
call_120 = security('call', 120, 1, -1)

bsPricer = BSPricer(model)
print bsPricer.price_vanilla(call_100)
print bsPricer.price_vanilla(call_120)

p = Portfolio()
p.add(call_100)
p.add(call_120)

print bsPricer.hedging_cost(p, 0.1)

fdmSolver = ExplicitFDMSolver(100, model)
fdmSolver.solve(p)
results = fdmSolver.get_results()
results.to_csv('vanilla_payoff.csv')
