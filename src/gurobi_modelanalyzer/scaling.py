import gurobipy as gp
import scipy.sparse
import numpy as np
from typing import List


import gurobi_modelanalyzer.common as common

class ScaledVar:
    """
    Wrapper around a Gurobi variable that provides access to unscaled values.
    """
    def __init__(self, gurobi_var, col_scaling_factor):
        self._var = gurobi_var
        self._col_scaling_factor = col_scaling_factor
    
    @property
    def X(self):
        """Scaled variable value"""
        return self._var.X
    
    @property
    def Xunsc(self):
        """Unscaled variable value: x = s * y"""
        return self._col_scaling_factor * self._var.X
    
    def __getattr__(self, name):
        """Forward all other attributes to the underlying Gurobi variable"""
        return getattr(self._var, name)


class ScaledModel(gp.Model):
    """
    A Gurobi model with scaling information attached.
    Allows easy access to unscaled variable values after optimization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._col_scaling = None
        self._row_scaling = None
        self._scaled_vars = None
    
    def getVarsUnscaled(self):
        """
        Get list of ScaledVar objects that have both X and Xunsc attributes.
        
        Returns:
        --------
        List[ScaledVar]
            List of wrapped variables with unscaling capabilities
        """
        if self._scaled_vars is None:
            if self._col_scaling is None:
                raise ValueError("No scaling information available.")
            
            gurobi_vars = self.getVars()
            col_scaling_diag = self._col_scaling.diagonal()
            
            self._scaled_vars = [
                ScaledVar(var, col_scaling_diag[i]) 
                for i, var in enumerate(gurobi_vars)
            ]
        
        return self._scaled_vars


class ModelData:
    """
    A reusable class to encapsulate Gurobi model data for easy manipulation and reconstruction.
    """
    
    def __init__(self, 
                constr_matrix: scipy.sparse.csr_matrix = None,
                rhs_vector: np.ndarray = None,
                constr_sense: List[str] = None,
                obj_vector: np.ndarray = None,
                ub_vector: np.ndarray = None,
                lb_vector: np.ndarray = None,
                var_types: List[str] = None,
                var_names: List[str] = None,
                constr_names: List[str] = None):
        
        self.constr_matrix = constr_matrix
        self.rhs_vector = rhs_vector
        self.constr_sense = constr_sense
        self.obj_vector = obj_vector
        self.ub_vector = ub_vector
        self.lb_vector = lb_vector
        self.var_types = var_types
        self.var_names = var_names
        self.constr_names = constr_names
        
        # # Compute non-zero elements per row and column
        # if constr_matrix is not None:
        #     # Convert to CSR format if not already
        #     csr_matrix = constr_matrix.tocsr()
            
        #     # Non-zero elements per row
        #     self.nnz_per_row = []
        #     self.num_nnz_per_row = []
        #     for i in range(csr_matrix.shape[0]):
        #         row_data = csr_matrix.getrow(i).data
        #         self.nnz_per_row.append(row_data.tolist())
        #         self.num_nnz_per_row.append(len(row_data))
            
        #     # Non-zero elements per column
        #     csc_matrix = constr_matrix.tocsc()
        #     self.nnz_per_col = []
        #     self.num_nnz_per_col = []
        #     for j in range(csc_matrix.shape[1]):
        #         col_data = csc_matrix.getcol(j).data
        #         self.nnz_per_col.append(col_data.tolist())
        #         self.num_nnz_per_col.append(len(col_data))
        # else:
        #     self.nnz_per_row = []
        #     self.num_nnz_per_row = []
        #     self.nnz_per_col = []
        #     self.num_nnz_per_col = []
        
    @classmethod
    def from_gurobi_model(cls, model):
        """
        Create ModelData from a Gurobi model.
        
        Parameters:
        -----------
        model : gp.Model
            Gurobi model to extract data from
            
        Returns:
        --------
        ModelData
            ModelData object containing all model information
        """
                
        return cls(
            constr_matrix=model.getA(),
            rhs_vector=np.array(model.getAttr("RHS")),
            constr_sense=model.getAttr("Sense"),
            obj_vector=np.array(model.getAttr("Obj")),
            ub_vector=np.array(model.getAttr("UB")),
            lb_vector=np.array(model.getAttr("LB")),
            var_types=model.getAttr("VType"),
            var_names=[var.VarName for var in model.getVars()],
            constr_names=[constr.ConstrName for constr in model.getConstrs()],
            )


def equilibration(constr_matrix: scipy.sparse.csr_matrix, ScalePasses: int, ScaleRelTol: float) -> scipy.sparse.csr_matrix:
    scaled_constr_matrix = constr_matrix
    # Initilize scaling matrices
    num_rows, num_cols = constr_matrix.shape
    row_scaling_total = scipy.sparse.eye(num_rows, format='csr')
    col_scaling_total = scipy.sparse.eye(num_cols, format='csr')
    
    for completed_scale_passes in range(ScalePasses):
        previous_matrix = scaled_constr_matrix.copy()
        
        # Compute row scaling for this iteration
        row_scaling_iter = scipy.sparse.eye(num_rows, format='csr')
        for i in range(num_rows):
            row_scaling_iter[i,i] = 1.0 / np.max(np.abs(scaled_constr_matrix.getrow(i).data))
        
        scaled_constr_matrix = row_scaling_iter @ scaled_constr_matrix
        row_scaling_total = row_scaling_iter @ row_scaling_total
        
        # Compute column scaling for this iteration
        col_scaling_iter = scipy.sparse.eye(num_cols, format='csr')
        for j in range(num_cols):
            col_scaling_iter[j,j] = 1.0 / np.max(np.abs(scaled_constr_matrix.getcol(j).data))
            
        scaled_constr_matrix = scaled_constr_matrix @ col_scaling_iter
        col_scaling_total = col_scaling_total @ col_scaling_iter
        
        # Check convergence
        norm_diff = scipy.sparse.linalg.norm(scaled_constr_matrix - previous_matrix, ord='fro')
        norm_prev = scipy.sparse.linalg.norm(previous_matrix, ord='fro')
        if norm_diff / norm_prev < ScaleRelTol:
            break
    
    return scaled_constr_matrix, row_scaling_total, col_scaling_total

def geometric_mean(constr_matrix: scipy.sparse.csr_matrix, ScalePasses: int, ScaleRelTol: float) -> scipy.sparse.csr_matrix:
    scaled_constr_matrix = constr_matrix
    # Initilize scaling matrices
    num_rows, num_cols = constr_matrix.shape
    row_scaling_total = scipy.sparse.eye(num_rows, format='csr')
    col_scaling_total = scipy.sparse.eye(num_cols, format='csr')
    
    for completed_scale_passes in range(ScalePasses):
        previous_matrix = scaled_constr_matrix.copy()
        
        # Compute row scaling for this iteration
        row_scaling_iter = scipy.sparse.eye(num_rows, format='csr')
        for i in range(num_rows):
            row_data = np.abs(scaled_constr_matrix.getrow(i).data)
            row_scaling_iter[i,i] = 1.0 / np.sqrt(np.min(row_data) * np.max(row_data))
        
        scaled_constr_matrix = row_scaling_iter @ scaled_constr_matrix
        row_scaling_total = row_scaling_iter @ row_scaling_total
        
        # Compute column scaling for this iteration
        col_scaling_iter = scipy.sparse.eye(num_cols, format='csr')
        for j in range(num_cols):
            col_data = np.abs(scaled_constr_matrix.getcol(j).data)
            col_scaling_iter[j,j] = 1.0 / np.sqrt(np.min(col_data) * np.max(col_data))
            
        scaled_constr_matrix = scaled_constr_matrix @ col_scaling_iter
        col_scaling_total = col_scaling_total @ col_scaling_iter
        
        # Check convergence
        norm_diff = scipy.sparse.linalg.norm(scaled_constr_matrix - previous_matrix, ord='fro')
        norm_prev = scipy.sparse.linalg.norm(previous_matrix, ord='fro')
        if norm_diff / norm_prev < ScaleRelTol:
            break

    return scaled_constr_matrix, row_scaling_total, col_scaling_total

def arithmetic_mean(constr_matrix: scipy.sparse.csr_matrix, ScalePasses: int, ScaleRelTol: float) -> scipy.sparse.csr_matrix:
    scaled_constr_matrix = constr_matrix.copy()
    # Initilize scaling matrices
    num_rows, num_cols = constr_matrix.shape
    row_scaling_total = scipy.sparse.eye(num_rows, format='csr')
    col_scaling_total = scipy.sparse.eye(num_cols, format='csr')
    
    for completed_scale_passes in range(ScalePasses):
        previous_matrix = scaled_constr_matrix.copy()
        
        # Compute row scaling for this iteration
        row_scaling_iter = scipy.sparse.eye(num_rows, format='csr')
        for i in range(num_rows):
            row_data = np.abs(scaled_constr_matrix.getrow(i).data)
            row_scaling_iter[i,i] = 1.0 / np.mean(row_data)
        
        scaled_constr_matrix = row_scaling_iter @ scaled_constr_matrix
        row_scaling_total = row_scaling_iter @ row_scaling_total
        
        # Compute column scaling for this iteration
        col_scaling_iter = scipy.sparse.eye(num_cols, format='csr')
        for j in range(num_cols):
            col_data = np.abs(scaled_constr_matrix.getcol(j).data)
            col_scaling_iter[j,j] = 1.0 / np.mean(col_data)
            
        scaled_constr_matrix = scaled_constr_matrix @ col_scaling_iter
        col_scaling_total = col_scaling_total @ col_scaling_iter
        
        # Check convergence
        norm_diff = scipy.sparse.linalg.norm(scaled_constr_matrix - previous_matrix, ord='fro')
        norm_prev = scipy.sparse.linalg.norm(previous_matrix, ord='fro')
        if norm_diff / norm_prev < ScaleRelTol:
            break


    return scaled_constr_matrix, row_scaling_total, col_scaling_total


def scale_model(model: gp.Model, method: str,
                ScalePasses=5, ScaleRelTol=1e-4) -> ScaledModel:
    # Get model data from original model
    model_data = ModelData.from_gurobi_model(model)
    
    # Compute scaled matrix and scaling factors
    if method == 'equilibration':
        scaled_matrix, row_scaling, col_scaling = equilibration(model_data.constr_matrix,
                                                                ScalePasses, ScaleRelTol)
    elif method == 'geometric_mean':
        scaled_matrix, row_scaling, col_scaling = geometric_mean(model_data.constr_matrix,
                                                                ScalePasses, ScaleRelTol)
    elif method == 'arithmetic_mean':
        scaled_matrix, row_scaling, col_scaling = arithmetic_mean(model_data.constr_matrix,
                                                                ScalePasses, ScaleRelTol)
    
    # Compute scaled data
    rhs_vector_scaled = row_scaling @ model_data.rhs_vector
    obj_vector_scaled = col_scaling @ model_data.obj_vector
    col_scaling_inv = scipy.sparse.diags(1.0 / col_scaling.diagonal())
    lb_vector_scaled = col_scaling_inv @ model_data.lb_vector
    ub_vector_scaled = col_scaling_inv @ model_data.ub_vector
    var_names_scaled = [name + "_scaled" for name in model_data.var_names]
    constr_names_scaled = [name + "_scaled" for name in model_data.constr_names]
    
   
    ## Create linear terms of ScaledModel with scaled data using matrix API
    model_scaled = ScaledModel(model.ModelName + "_scaled")
    
    # Add variables with scaled bounds and objective
    vars_list = []
    for i in range(len(model_data.var_names)):
        var = model_scaled.addVar(
            lb=lb_vector_scaled[i],
            ub=ub_vector_scaled[i],
            obj=obj_vector_scaled[i],
            vtype=model_data.var_types[i],
            name=var_names_scaled[i]
        )
        vars_list.append(var)
    
    model_scaled.update()
    
    # Add scaled constraints using matrix API
    model_scaled.addMConstr(
        scaled_matrix,
        vars_list,
        model_data.constr_sense,
        rhs_vector_scaled,
        constr_names_scaled
    )
    
    model_scaled.update()
    
    # Store scaling matrices
    model_scaled._col_scaling = col_scaling
    model_scaled._row_scaling = row_scaling
    
    ## Scale quadratic objective if present
    if model.isQP or model.isQCP:
        # Extract quadratic objective matrix
        Q_objective_matrix = model.getQ()
        # Apply column scaling
        Q_objective_matrix_scaled = col_scaling @ Q_objective_matrix @ col_scaling
        # Set quadratic objective in scaled model
        model_scaled.setMObjective(
            Q_objective_matrix_scaled, obj_vector_scaled, 0.0, sense=model.ModelSense)
    ## Scale quadratic constraints if present
    if model.isQCP:
        quad_scaling_factors = []
        for qconstr in model.getQConstrs():
            # Extract data from constraint
            Q, q = model.getQCMatrices(qconstr)
            Q = col_scaling @ Q @ col_scaling
            q = col_scaling @ q
            print(q)
            rhs  = qconstr.QCRHS
            sense = qconstr.QCSense
            name = qconstr.QCName + "_scaled"
            # Compute scaling factor for constraint
            scaling_factor = 1.0 / max(
                scipy.sparse.linalg.norm(Q, ord='fro'), # ||S*Q*S||_F
                scipy.sparse.linalg.norm(q),     # ||S*q||_2
                abs(rhs),                               # |rhs|     
                1.0)
            quad_scaling_factors.append(scaling_factor)
            Q = scaling_factor * Q
            q = scaling_factor * q
            rhs = scaling_factor * rhs
            # Add scaled quadratic constraint
            model_scaled.addMQConstr(
                Q, q.toarray().flatten(), sense, rhs, name=name)
        model_scaled.update()
        # Add scaling factors to scaled model
        model_scaled._quad_scaling_factors = quad_scaling_factors
    return model_scaled