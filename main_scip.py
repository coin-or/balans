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