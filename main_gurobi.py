import gurobipy as grb
try:
    # Specific initialization (discard if not applicable)
    from gurobi_onboarder import init_gurobi
    gurobi_venv, GUROBI_FOUND = init_gurobi.initialize_gurobi()
    if gurobi_venv is not None:
        gurobi_venv.setParam("OutputFlag", 1)
        gurobi_venv.setParam("LogToConsole", 0)
        gurobi_venv.setParam("LogFile", "temp.log")
except Exception:
    gurobi_venv = None

# Instance
instance_path = "tests/data/noswot.mps"
model = grb.read(f'{instance_path}', env=gurobi_venv)
model.setParam("TimeLimit", 600)

# Solve
model.optimize()

# Solution
print("Best objective:", model.getObjective())
# for v in model.getVars():
#     print('%s %g' % (v.varName, v.x))

