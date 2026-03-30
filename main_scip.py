from pyscipopt import Model

# Model
model = Model("SCIP")

# Instance
instance_path = "tests/data/noswot.mps"
model.readProblem(instance_path)
model.setParam('limits/time', 600) # seconds

# Solve
model.optimize()

# Solution
solution = model.getBestSol()
# print("Best solution:", solution)
print("Best objective:", model.getObjVal())

# Check if a problem is maximization or minization
# python -c "import pyscipopt; m=pyscipopt.Model(); m.readProblem('tests/data/pk1.mps'); print(m.getObjectiveSense())"