import gurobipy as gp
from gurobipy import GRB
from gurobi_modelanalyzer.scaling import scale_model

# 1. Badly Scaled LP (coefficient range spans ~10 orders of magnitude)
model_lp = gp.Model("BadlyScaled_LP")
model_lp.setParam("OutputFlag", 0)
x1 = model_lp.addVar(lb=1e-3, ub=1e-2, name="x1")
x2 = model_lp.addVar(lb=0, ub=100, name="x2")
x3 = model_lp.addVar(lb=0, ub=1e4, name="x3")

model_lp.setObjective(1e5*x1 + 1*x2 + 1e-5*x3, GRB.MINIMIZE)
model_lp.addConstr(1e4*x1 + 1e-2*x2 + 1*x3 >= 10)
model_lp.addConstr(1e-3*x1 + 1e3*x2 + 1e-4*x3 <= 1e5)
model_lp.addConstr(x2 >= 1)

# Scale the LP model
model_lp_scaled = scale_model(model_lp, method='arithmetic_mean', ScalePasses=10)
model_lp_scaled.setParam("OutputFlag", 0)

# Solve both models
model_lp.optimize()
model_lp_scaled.optimize()

# Print results
print("="*60)
print("LP Example")
print("="*60)
print("Original solution:")
for var in model_lp.getVars():
    print(f"  {var.VarName} = {var.X:.6e}")
print(f"  Objective = {model_lp.ObjVal:.6e}")

print("\nUnscaled solution from scaled model:")
for var in model_lp_scaled.getVarsUnscaled():
    print(f"  {var.VarName.replace('_scaled', '')} = {var.Xunsc:.6e}")
print(f"  Objective = {model_lp.ObjVal:.6e}")

# Compute violations
model_lp_scaled.ComputeUnscVio(model_lp)
print(f"\nViolations:")
print(f"  Max constraint violation: {model_lp_scaled.MaxUnscConstrVio:.6e}")
print(f"  Max bound violation: {model_lp_scaled.MaxUnscBoundVio:.6e}")
print(f"  Max total violation: {model_lp_scaled.MaxUnscVio:.6e}")
print()

# 2. Badly Scaled QP (coefficient range spans ~10 orders of magnitude)
model_qp = gp.Model("BadlyScaled_QP")
model_qp.setParam("OutputFlag", 0)
x = model_qp.addVars(4, lb=[1e-4, 0, 0, 0], ub=[1e-3, 10, 1e3, 1e4], name="x")

obj = 0.5*1e6*x[0]*x[0] + 0.5*1e2*x[1]*x[1] + 0.5*1*x[2]*x[2] + 0.5*1e-4*x[3]*x[3]
obj += 1e-2*x[0] + 1*x[1] + 1e3*x[2] + 1e4*x[3]
model_qp.setObjective(obj, GRB.MINIMIZE)

model_qp.addConstr(1e5*x[0] + 1e-2*x[1] + 1*x[2] + 1e-3*x[3] >= 0.1)
model_qp.addConstr(1e-3*x[0] + 1e3*x[1] + 1e-1*x[2] + 1*x[3] <= 1e4)
model_qp.addConstr(x[1] + x[2] >= 1)
model_qp.update()

# Scale QP model
model_qp_scaled = scale_model(model_qp, method='equilibration', ScalePasses=5)
model_qp_scaled.setParam("OutputFlag", 0)

# Solve both models
model_qp.optimize()
model_qp_scaled.optimize()

# Print results
print("="*60)
print("QP Example")
print("="*60)
print("Original solution:")
for var in model_qp.getVars():
    print(f"  {var.VarName} = {var.X:.6e}")
print(f"  Objective = {model_qp.ObjVal:.6e}")

print("\nUnscaled solution from scaled model:")
for var in model_qp_scaled.getVarsUnscaled():
    print(f"  {var.VarName.replace('_scaled', '')} = {var.Xunsc:.6e}")
print(f"  Objective = {model_qp.ObjVal:.6e}")

# Compute violations
model_qp_scaled.ComputeUnscVio(model_qp)
print(f"\nViolations:")
print(f"  Max constraint violation: {model_qp_scaled.MaxUnscConstrVio:.6e}")
print(f"  Max bound violation: {model_qp_scaled.MaxUnscBoundVio:.6e}")
print(f"  Max total violation: {model_qp_scaled.MaxUnscVio:.6e}")
print()

# 3. Badly Scaled QCP (linear objective, quadratic constraints, ~10 order range)
model_qcp_lin = gp.Model("BadlyScaled_QCP_LinearObj")
model_qcp_lin.setParam("OutputFlag", 0)
x = model_qcp_lin.addVars(3, lb=[0, 0, 0], ub=[1e-2, 1e2, 1e3], name="x")

model_qcp_lin.setObjective(1e4*x[0] + 1*x[1] + 1e-4*x[2], GRB.MINIMIZE)

# Quadratic constraints with more moderate scaling
model_qcp_lin.addConstr(1e6*x[0]*x[0] + x[1]*x[1] + 1e-6*x[2]*x[2] <= 100)
model_qcp_lin.addConstr(x[0] + x[1] + 1e-3*x[2] >= 1)
model_qcp_lin.addConstr(x[1] >= 0.5)

# Scale QCP model
model_qcp_lin_scaled = scale_model(model_qcp_lin, method='equilibration', ScalePasses=5)
model_qcp_lin_scaled.setParam("OutputFlag", 0)

# Solve both models
model_qcp_lin.optimize()
model_qcp_lin_scaled.optimize()

# Print results
print("="*60)
print("QCP Example (Linear Objective)")
print("="*60)
if model_qcp_lin.Status == GRB.OPTIMAL:
    print("Original solution:")
    for var in model_qcp_lin.getVars():
        print(f"  {var.VarName} = {var.X:.6e}")
    print(f"  Objective = {model_qcp_lin.ObjVal:.6e}")

    print("\nUnscaled solution from scaled model:")
    for var in model_qcp_lin_scaled.getVarsUnscaled():
        print(f"  {var.VarName.replace('_scaled', '')} = {var.Xunsc:.6e}")
    print(f"  Objective = {model_qcp_lin.ObjVal:.6e}")
    
    # Compute violations
    model_qcp_lin_scaled.ComputeUnscVio(model_qcp_lin)
    print(f"\nViolations:")
    print(f"  Max constraint violation: {model_qcp_lin_scaled.MaxUnscConstrVio:.6e}")
    print(f"  Max bound violation: {model_qcp_lin_scaled.MaxUnscBoundVio:.6e}")
    print(f"  Max total violation: {model_qcp_lin_scaled.MaxUnscVio:.6e}")
else:
    print(f"Model status: {model_qcp_lin.Status} (not optimal)")
print()


# 4. Badly Scaled QCP (quadratic objective, quadratic constraints, ~10 order range)
model_qcp_quad = gp.Model("BadlyScaled_QCP_QuadObj")
model_qcp_quad.setParam("OutputFlag", 0)
x = model_qcp_quad.addVars(3, lb=[0, 0, 0], ub=[1e-2, 1e2, 1e3], name="x")

obj = 0.5*1e5*x[0]*x[0] + 0.5*x[1]*x[1] + 0.5*1e-5*x[2]*x[2]
obj += 1e2*x[0] + x[1] + 1e-2*x[2]
model_qcp_quad.setObjective(obj, GRB.MINIMIZE)

# Quadratic constraints
model_qcp_quad.addConstr(1e7*x[0]*x[0] + x[1]*x[1] + 1e-7*x[2]*x[2] <= 100)
model_qcp_quad.addConstr(x[0] + x[1] + 1e-3*x[2] >= 1)
model_qcp_quad.addConstr(x[1] >= 0.5)

# Scale QCP model
model_qcp_quad_scaled = scale_model(model_qcp_quad, method='equilibration', ScalePasses=5)
model_qcp_quad_scaled.setParam("OutputFlag", 0)

# Solve both models
model_qcp_quad.optimize()
model_qcp_quad_scaled.optimize()

# Print results
print("="*60)
print("QCP Example (Quadratic Objective)")
print("="*60)
if model_qcp_quad.Status == GRB.OPTIMAL:
    print("Original solution:")
    for var in model_qcp_quad.getVars():
        print(f"  {var.VarName} = {var.X:.6e}")
    print(f"  Objective = {model_qcp_quad.ObjVal:.6e}")

    print("\nUnscaled solution from scaled model:")
    for var in model_qcp_quad_scaled.getVarsUnscaled():
        print(f"  {var.VarName.replace('_scaled', '')} = {var.Xunsc:.6e}")
    print(f"  Objective = {model_qcp_quad.ObjVal:.6e}")
    
    # Compute violations
    model_qcp_quad_scaled.ComputeUnscVio(model_qcp_quad)
    print(f"\nViolations:")
    print(f"  Max constraint violation: {model_qcp_quad_scaled.MaxUnscConstrVio:.6e}")
    print(f"  Max bound violation: {model_qcp_quad_scaled.MaxUnscBoundVio:.6e}")
    print(f"  Max total violation: {model_qcp_quad_scaled.MaxUnscVio:.6e}")
else:
    print(f"Model status: {model_qcp_quad.Status} (not optimal)")
print()
