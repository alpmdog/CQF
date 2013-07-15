from uvm import *

model = modelparams(0.02, 100, 0.1, 0.3)
digital_call = Security('digital call', 110, 1, 1)
call_100 = Security('call', 100, 1, -0.05)
call_120 = Security('call', 120, 1, 0.05)

p = Portfolio()
p.add(digital_call)
p.add(call_100)
p.add(call_120)

optimized_params = optimize_portfolio(p, model)
print optimized_params
