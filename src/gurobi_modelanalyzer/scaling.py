import gurobipy as gp
from gurobipy import GRB
import scipy.sparse
import numpy as np
from typing import List, Tuple, Dict, Union
import warnings
from joblib import Parallel, delayed


import gurobi_modelanalyzer.common as common

class ScaledVar:
    """
    Wrapper around a Gurobi variable that provides access to unscaled values and bound violations.
    """
    def __init__(self, gurobi_var, col_scaling_factor):
        self._var = gurobi_var
        self._col_scaling_factor = col_scaling_factor
        self.UnscBoundViolation = None  # Will be set by ComputeUnscVio
    
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


class ScaledConstr:
    """
    Wrapper around a Gurobi constraint that provides access to unscaled violations.
    """
    def __init__(self, gurobi_constr):
        self._constr = gurobi_constr
        self._unsc_violation = None
    
    @property
    def UnscViolation(self):
        """Unscaled constraint violation"""
        return self._unsc_violation
    
    @UnscViolation.setter
    def UnscViolation(self, value):
        """Set unscaled constraint violation"""
        self._unsc_violation = value
    
    def __getattr__(self, name):
        """Forward all other attributes to the underlying Gurobi constraint"""
        return getattr(self._constr, name)


class ScaledQConstr:
    """
    Wrapper around a Gurobi quadratic constraint that provides access to unscaled violations.
    """
    def __init__(self, gurobi_qconstr):
        self._qconstr = gurobi_qconstr
        self._unsc_violation = None
    
    @property
    def UnscViolation(self):
        """Unscaled quadratic constraint violation"""
        return self._unsc_violation
    
    @UnscViolation.setter
    def UnscViolation(self, value):
        """Set unscaled quadratic constraint violation"""
        self._unsc_violation = value
    
    def __getattr__(self, name):
        """Forward all other attributes to the underlying Gurobi quadratic constraint"""
        return getattr(self._qconstr, name)


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
        self._scaled_constrs = None
        self._scaled_qconstrs = None
        self._constraint_violations = None
        self._max_unsc_vio = None
        self._original_model = None
    
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
    
    def getConstrsUnscaled(self):
        """
        Get list of ScaledConstr objects with UnscViolation attributes.
        
        Returns:
        --------
        List[ScaledConstr]
            List of wrapped constraints with violation tracking
        """
        if self._scaled_constrs is None:
            if self._original_model is None:
                raise ValueError("Original model not stored. Cannot access constraint wrappers.")
            
            orig_constrs = self._original_model.getConstrs()
            self._scaled_constrs = [ScaledConstr(c) for c in orig_constrs]
        
        return self._scaled_constrs
    
    def getQConstrsUnscaled(self):
        """
        Get list of ScaledQConstr objects with UnscViolation attributes.
        
        Returns:
        --------
        List[ScaledQConstr]
            List of wrapped quadratic constraints with violation tracking
        """
        if self._scaled_qconstrs is None:
            if self._original_model is None:
                raise ValueError("Original model not stored. Cannot access constraint wrappers.")
            
            orig_qconstrs = self._original_model.getQConstrs()
            self._scaled_qconstrs = [ScaledQConstr(qc) for qc in orig_qconstrs]
        
        return self._scaled_qconstrs
    
    def ComputeUnscVio(self, original_model):
        """
        Compute unscaled constraint and bound violations using the unscaled variable values.
        Stores violations in constraint wrapper objects and variable wrapper objects,
        and tracks maximum violation.
        
        After calling this method, violations can be accessed via:
        - constraint.UnscViolation for each constraint
        - var.UnscBoundViolation for each variable's bound violation
        - model_scaled.MaxUnscVio for the maximum violation across all constraints and bounds
        - model_scaled.MaxUnscConstrVio for max constraint violation only
        - model_scaled.MaxUnscBoundVio for max bound violation only
        
        Parameters:
        -----------
        original_model : gp.Model
            The original unscaled model with same structure as this scaled model
        """
        # Store reference to original model
        self._original_model = original_model
        
        # Get unscaled solution values
        unscaled_vars = self.getVarsUnscaled()
        unscaled_values = [var.Xunsc for var in unscaled_vars]
        
        # Compute constraint and bound violations using the original model
        violations = compute_constraint_violations(original_model, unscaled_values)
        
        # Store violations in dictionaries
        self._constraint_violations = violations['constraints']
        self._bound_violations = violations['bounds']
        
        # Store violations in constraint wrappers
        scaled_constrs = self.getConstrsUnscaled()
        for scaled_constr in scaled_constrs:
            constr_name = scaled_constr.ConstrName
            scaled_constr.UnscViolation = self._constraint_violations.get(constr_name, 0.0)
        
        # Store violations in quadratic constraint wrappers
        if original_model.NumQConstrs > 0:
            scaled_qconstrs = self.getQConstrsUnscaled()
            for scaled_qconstr in scaled_qconstrs:
                qconstr_name = scaled_qconstr.QCName
                scaled_qconstr.UnscViolation = self._constraint_violations.get(qconstr_name, 0.0)
        
        # Store bound violations in variable wrappers
        for i, var in enumerate(unscaled_vars):
            var_name = var.VarName.replace('_scaled', '')
            var.UnscBoundViolation = self._bound_violations.get(var_name, 0.0)
        
        # Compute and store maximum violations
        all_constraint_vios = list(self._constraint_violations.values())
        all_bound_vios = list(self._bound_violations.values())
        
        self._max_unsc_constr_vio = max(all_constraint_vios) if all_constraint_vios else 0.0
        self._max_unsc_bound_vio = max(all_bound_vios) if all_bound_vios else 0.0
        self._max_unsc_vio = max(self._max_unsc_constr_vio, self._max_unsc_bound_vio)
    
    @property
    def MaxUnscVio(self):
        """
        Get the maximum unscaled violation across all constraints and bounds.
        
        Returns:
        --------
        float
            Maximum violation, or None if not computed
        """
        return getattr(self, '_max_unsc_vio', None)
    
    @property
    def MaxUnscConstrVio(self):
        """
        Get the maximum unscaled constraint violation (linear and quadratic constraints only).
        
        Returns:
        --------
        float
            Maximum constraint violation, or None if not computed
        """
        return getattr(self, '_max_unsc_constr_vio', None)
    
    @property
    def MaxUnscBoundVio(self):
        """
        Get the maximum unscaled bound violation.
        
        Returns:
        --------
        float
            Maximum bound violation, or None if not computed
        """
        return getattr(self, '_max_unsc_bound_vio', None)


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


def _compute_row_scaling_factor(i: int, scaled_matrix: scipy.sparse.csr_matrix, method: str) -> float:
    """
    Compute scaling factor for a single row.
    
    Parameters:
    -----------
    i : int
        Row index
    scaled_matrix : scipy.sparse.csr_matrix
        Current scaled matrix
    method : str
        Scaling method ('equilibration', 'arithmetic_mean', or 'geometric_mean')
        
    Returns:
    --------
    float
        Scaling factor for the row
    """
    row_data = np.abs(scaled_matrix.getrow(i).data)
    if len(row_data) == 0:
        return 1.0
    
    if method in ['equilibration', 'arithmetic_mean']:
        return 1.0 / np.mean(row_data)
    elif method == 'geometric_mean':
        return 1.0 / np.sqrt(np.min(row_data) * np.max(row_data))
    else:
        return 1.0


def _compute_col_scaling_factor(j: int, scaled_matrix: scipy.sparse.csr_matrix, method: str) -> float:
    """
    Compute scaling factor for a single column.
    
    Parameters:
    -----------
    j : int
        Column index
    scaled_matrix : scipy.sparse.csr_matrix
        Current scaled matrix
    method : str
        Scaling method ('equilibration', 'arithmetic_mean', or 'geometric_mean')
        
    Returns:
    --------
    float
        Scaling factor for the column
    """
    col_data = np.abs(scaled_matrix.getcol(j).data)
    if len(col_data) == 0:
        return 1.0
    
    if method == 'equilibration':
        return 1.0 / np.max(col_data)
    elif method == 'arithmetic_mean':
        return 1.0 / np.mean(col_data)
    elif method == 'geometric_mean':
        return 1.0 / np.sqrt(np.min(col_data) * np.max(col_data))
    else:
        return 1.0


def _compute_kkt_col_scaling_factor(i: int, KKT_matrix: scipy.sparse.csr_matrix, 
                                    num_cols: int, cols_to_scale: List[int],
                                    ScalingLB: float, ScalingUB: float) -> float:
    """
    Compute diagonal scaling factor for a single KKT matrix column.
    
    Parameters:
    -----------
    i : int
        Column index in KKT matrix
    KKT_matrix : scipy.sparse.csr_matrix
        Current KKT matrix
    num_cols : int
        Number of variables (to distinguish variables from constraints)
    cols_to_scale : List[int]
        Indices of columns to scale
    ScalingLB : float
        Lower bound for scaling factors
    ScalingUB : float
        Upper bound for scaling factors
        
    Returns:
    --------
    float
        Scaling factor for the KKT matrix column
    """
    if i < num_cols and i not in cols_to_scale:
        return 1.0
    
    col_data = np.abs(KKT_matrix.getcol(i).data)
    if len(col_data) == 0:
        return 1.0
    
    max_val = np.max(col_data)
    scaling_factor = 1.0 / np.sqrt(max_val)
    return np.clip(scaling_factor, ScalingLB, ScalingUB)


def _scale_single_qconstr(qconstr: gp.QConstr, model: gp.Model, 
                         col_scaling: scipy.sparse.csr_matrix) -> Tuple[scipy.sparse.csr_matrix, np.ndarray, str, float, float, str]:
    """
    Compute scaling for a single quadratic constraint.
    
    Parameters:
    -----------
    qconstr : gp.QConstr
        Quadratic constraint to scale
    model : gp.Model
        Original model containing the constraint
    col_scaling : scipy.sparse.csr_matrix
        Column scaling matrix
        
    Returns:
    --------
    Tuple containing:
        - Qc_scaled: Scaled quadratic matrix
        - q_scaled: Scaled linear vector
        - sense: Constraint sense
        - rhs_scaled: Scaled RHS
        - scaling_factor: Computed scaling factor
        - name: Constraint name
    """
    # Extract data from constraint
    Qc, q = model.getQCMatrices(qconstr)
    Qc = col_scaling @ Qc @ col_scaling
    q = col_scaling @ q
    rhs = qconstr.QCRHS
    sense = qconstr.QCSense
    name = qconstr.QCName + "_scaled"
    
    # Qc is upper triangular from Gurobi, make it symmetric for norm calculation
    Qc_full = Qc + Qc.T - scipy.sparse.diags(Qc.diagonal())
    
    # Compute scaling factor for constraint
    scaling_factor = 1.0 / max(
        scipy.sparse.linalg.norm(Qc_full, ord='fro'),  # ||S*Q*S||_F (full symmetric)
        scipy.sparse.linalg.norm(q),                   # ||S*q||_2
        abs(rhs),                                      # |rhs|
        1.0)
    
    Qc_scaled = scaling_factor * Qc
    q_scaled = scaling_factor * q
    rhs_scaled = scaling_factor * rhs
    
    return Qc_scaled, q_scaled, sense, rhs_scaled, scaling_factor, name


def threshold_small_coefficients(data: Union[scipy.sparse.spmatrix, np.ndarray], 
                                value_threshold: float = 1e-13) -> Union[scipy.sparse.csr_matrix, np.ndarray]:
    """
    Set coefficients below threshold to zero.
    Works for both sparse matrices and numpy arrays.
    
    Parameters:
    -----------
    data : scipy.sparse matrix or np.ndarray
        The data to threshold
    value_threshold : float, optional
        Absolute value threshold below which coefficients are set to zero (default: 1e-13)
        
    Returns:
    --------
    scipy.sparse.csr_matrix or np.ndarray
        Thresholded data in the same format as input
    """
    if scipy.sparse.issparse(data):
        # For sparse matrices
        data_lil = data.tolil()
        for i in range(data_lil.shape[0]):
            for j, val in zip(data_lil.rows[i], data_lil.data[i]):
                if abs(val) < value_threshold:
                    data_lil[i, j] = 0.0
        # Convert to csr and eliminate zeros
        return data_lil.tocsr()  # tocsr() automatically eliminates zeros
    else:
        # For numpy arrays
        result = data.copy()
        result[np.abs(result) < value_threshold] = 0.0
        return result

def equilibration(constr_matrix: scipy.sparse.csr_matrix, 
                  cols_to_scale: List[int], 
                  ScalePasses: int, 
                  ScaleRelTol: float,
                  n_jobs: int = -1) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Scale constraint matrix using equilibration method.
    
    Applies iterative row and column scaling using mean-based equilibration.
    Row scaling uses the mean of absolute row values, column scaling uses
    the maximum of absolute column values.
    
    Parameters:
    -----------
    constr_matrix : scipy.sparse.csr_matrix
        The constraint matrix to scale
    cols_to_scale : List[int]
        List of column indices to scale (typically continuous variables)
    ScalePasses : int
        Maximum number of scaling iterations
    ScaleRelTol : float
        Relative tolerance for convergence check
    n_jobs : int, optional
        Number of parallel threads to use. -1 means use all processors (default: -1)
        
    Returns:
    --------
    Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]
        - scaled_constr_matrix: The scaled constraint matrix
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
    """
    scaled_constr_matrix = constr_matrix
    # Initialize scaling matrices
    num_rows, num_cols = constr_matrix.shape
    row_scaling_total = scipy.sparse.eye(num_rows, format='csr')
    col_scaling_total = scipy.sparse.eye(num_cols, format='csr')
    
    for completed_scale_passes in range(ScalePasses):
        previous_matrix = scaled_constr_matrix.copy()
        
        # Compute row scaling in parallel
        row_factors = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_compute_row_scaling_factor)(i, scaled_constr_matrix, 'equilibration')
            for i in range(num_rows)
        )
        row_scaling_iter = scipy.sparse.diags(row_factors)
        scaled_constr_matrix = row_scaling_iter @ scaled_constr_matrix
        row_scaling_total = row_scaling_iter @ row_scaling_total
        
        # Compute column scaling in parallel
        col_factors = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_compute_col_scaling_factor)(j, scaled_constr_matrix, 'equilibration')
            for j in cols_to_scale
        )
        # Build full column scaling (1.0 for non-scaled columns)
        col_factors_full = np.ones(num_cols)
        for idx, j in enumerate(cols_to_scale):
            col_factors_full[j] = col_factors[idx]
        
        col_scaling_iter = scipy.sparse.diags(col_factors_full)
        scaled_constr_matrix = scaled_constr_matrix @ col_scaling_iter
        col_scaling_total = col_scaling_total @ col_scaling_iter
        
        # Check convergence
        norm_diff = scipy.sparse.linalg.norm(scaled_constr_matrix - previous_matrix, ord='fro')
        norm_prev = scipy.sparse.linalg.norm(previous_matrix, ord='fro')
        if norm_diff / norm_prev < ScaleRelTol:
            break
    
    return scaled_constr_matrix, row_scaling_total, col_scaling_total

def geometric_mean(constr_matrix: scipy.sparse.csr_matrix, 
                   cols_to_scale: List[int], 
                   ScalePasses: int, 
                   ScaleRelTol: float,
                   n_jobs: int = -1) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Scale constraint matrix using geometric mean method.
    
    Applies iterative row and column scaling using geometric mean of
    minimum and maximum absolute values in each row/column.
    
    Parameters:
    -----------
    constr_matrix : scipy.sparse.csr_matrix
        The constraint matrix to scale
    cols_to_scale : List[int]
        List of column indices to scale (typically continuous variables)
    ScalePasses : int
        Maximum number of scaling iterations
    ScaleRelTol : float
        Relative tolerance for convergence check
    n_jobs : int, optional
        Number of parallel threads to use. -1 means use all processors (default: -1)
        
    Returns:
    --------
    Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]
        - scaled_constr_matrix: The scaled constraint matrix
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
    """
    scaled_constr_matrix = constr_matrix
    # Initialize scaling matrices
    num_rows, num_cols = constr_matrix.shape
    row_scaling_total = scipy.sparse.eye(num_rows, format='csr')
    col_scaling_total = scipy.sparse.eye(num_cols, format='csr')
    
    for completed_scale_passes in range(ScalePasses):
        previous_matrix = scaled_constr_matrix.copy()
        
        # Compute row scaling in parallel
        row_factors = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_compute_row_scaling_factor)(i, scaled_constr_matrix, 'geometric_mean')
            for i in range(num_rows)
        )
        row_scaling_iter = scipy.sparse.diags(row_factors)
        scaled_constr_matrix = row_scaling_iter @ scaled_constr_matrix
        row_scaling_total = row_scaling_iter @ row_scaling_total
        
        # Compute column scaling in parallel
        col_factors = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_compute_col_scaling_factor)(j, scaled_constr_matrix, 'geometric_mean')
            for j in cols_to_scale
        )
        # Build full column scaling (1.0 for non-scaled columns)
        col_factors_full = np.ones(num_cols)
        for idx, j in enumerate(cols_to_scale):
            col_factors_full[j] = col_factors[idx]
        
        col_scaling_iter = scipy.sparse.diags(col_factors_full)
        scaled_constr_matrix = scaled_constr_matrix @ col_scaling_iter
        col_scaling_total = col_scaling_total @ col_scaling_iter
        
        # Check convergence
        norm_diff = scipy.sparse.linalg.norm(scaled_constr_matrix - previous_matrix, ord='fro')
        norm_prev = scipy.sparse.linalg.norm(previous_matrix, ord='fro')
        if norm_diff / norm_prev < ScaleRelTol:
            break

    return scaled_constr_matrix, row_scaling_total, col_scaling_total

def arithmetic_mean(constr_matrix: scipy.sparse.csr_matrix, 
                    cols_to_scale: List[int], 
                    ScalePasses: int, 
                    ScaleRelTol: float,
                    n_jobs: int = -1) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Scale constraint matrix using arithmetic mean method.
    
    Applies iterative row and column scaling using arithmetic mean of
    absolute values in each row/column.
    
    Parameters:
    -----------
    constr_matrix : scipy.sparse.csr_matrix
        The constraint matrix to scale
    cols_to_scale : List[int]
        List of column indices to scale (typically continuous variables)
    ScalePasses : int
        Maximum number of scaling iterations
    ScaleRelTol : float
        Relative tolerance for convergence check
    n_jobs : int, optional
        Number of parallel threads to use. -1 means use all processors (default: -1)
        
    Returns:
    --------
    Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]
        - scaled_constr_matrix: The scaled constraint matrix
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
    """
    scaled_constr_matrix = constr_matrix.copy()
    # Initialize scaling matrices
    num_rows, num_cols = constr_matrix.shape
    row_scaling_total = scipy.sparse.eye(num_rows, format='csr')
    col_scaling_total = scipy.sparse.eye(num_cols, format='csr')
    
    for completed_scale_passes in range(ScalePasses):
        previous_matrix = scaled_constr_matrix.copy()
        
        # Compute row scaling in parallel
        row_factors = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_compute_row_scaling_factor)(i, scaled_constr_matrix, 'arithmetic_mean')
            for i in range(num_rows)
        )
        row_scaling_iter = scipy.sparse.diags(row_factors)
        scaled_constr_matrix = row_scaling_iter @ scaled_constr_matrix
        row_scaling_total = row_scaling_iter @ row_scaling_total
        
        # Compute column scaling in parallel
        col_factors = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_compute_col_scaling_factor)(j, scaled_constr_matrix, 'arithmetic_mean')
            for j in cols_to_scale
        )
        # Build full column scaling (1.0 for non-scaled columns)
        col_factors_full = np.ones(num_cols)
        for idx, j in enumerate(cols_to_scale):
            col_factors_full[j] = col_factors[idx]
        
        col_scaling_iter = scipy.sparse.diags(col_factors_full)
        scaled_constr_matrix = scaled_constr_matrix @ col_scaling_iter
        col_scaling_total = col_scaling_total @ col_scaling_iter
        
        # Check convergence
        norm_diff = scipy.sparse.linalg.norm(scaled_constr_matrix - previous_matrix, ord='fro')
        norm_prev = scipy.sparse.linalg.norm(previous_matrix, ord='fro')
        if norm_diff / norm_prev < ScaleRelTol:
            break


    return scaled_constr_matrix, row_scaling_total, col_scaling_total



def quad_equilibration(constr_matrix: scipy.sparse.csr_matrix, 
                       obj_vector: np.ndarray, 
                       Q_matrix: scipy.sparse.coo_matrix,
                       cols_to_scale: List[int], 
                       ScalePasses: int, 
                       ScaleRelTol: float,
                       ScalingLB: float = 1e-8, 
                       ScalingUB: float = 1e8,
                       n_jobs: int = -1) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Scale quadratic program using KKT-based equilibration method.
    
    Scales the constraint matrix, quadratic objective matrix (Q), and linear
    objective vector jointly by building a KKT matrix and applying equilibration
    to both the variable/constraint scaling and the objective scaling.
    
    Parameters:
    -----------
    constr_matrix : scipy.sparse.csr_matrix
        The constraint matrix to scale
    obj_vector : np.ndarray
        Linear objective coefficient vector
    Q_matrix : scipy.sparse.coo_matrix
        Quadratic objective matrix (Hessian)
    cols_to_scale : List[int]
        List of column indices to scale (typically continuous variables)
    ScalePasses : int
        Maximum number of scaling iterations
    ScaleRelTol : float
        Relative tolerance for convergence check
    ScalingLB : float, optional
        Lower bound for scaling factors (default: 1e-8)
    ScalingUB : float, optional
        Upper bound for scaling factors (default: 1e8)
    n_jobs : int, optional
        Number of parallel threads to use. -1 means use all processors (default: -1)
        
    Returns:
    --------
    Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]
        - scaled_constr_matrix: The scaled constraint matrix
        - scaled_Q_matrix: The scaled quadratic objective matrix
        - scaled_obj_vector: The scaled linear objective vector
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
    """
    scaled_constr_matrix = constr_matrix.copy()
    scaled_Q_matrix = Q_matrix.copy()
    scaled_obj_vector = obj_vector.copy()
    
    # Initialize scaling matrices
    num_rows, num_cols = constr_matrix.shape
    diagonal_scaling_total = scipy.sparse.eye(num_rows + num_cols, format='csr')
    obj_scaling_factor_total = 1.0
    zero_block = scipy.sparse.csr_matrix((num_rows, num_rows))
    
    for completed_scale_passes in range(ScalePasses):
        previous_constr_matrix = scaled_constr_matrix.copy()
        previous_Q_matrix = scaled_Q_matrix.copy()
        previous_obj_vector = scaled_obj_vector.copy()
        
        # Build KKT matrix from CURRENT scaled matrices
        KKT_matrix = scipy.sparse.bmat([
            [scaled_Q_matrix, scaled_constr_matrix.T],
            [scaled_constr_matrix, zero_block]
        ]).tocsr()
        
        # Compute diagonal scaling factors for this iteration in parallel (M equilibration)
        diagonal_factors = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_compute_kkt_col_scaling_factor)(i, KKT_matrix, num_cols, cols_to_scale, ScalingLB, ScalingUB)
            for i in range(num_rows + num_cols)
        )
        diagonal_scaling_iter = scipy.sparse.diags(diagonal_factors)
        
        # Extract column and row scaling from THIS iteration
        col_scaling_iter = scipy.sparse.diags(diagonal_scaling_iter.diagonal()[:num_cols])
        row_scaling_iter = scipy.sparse.diags(diagonal_scaling_iter.diagonal()[num_cols:])
        
        # Apply M equilibration scaling
        scaled_constr_matrix = row_scaling_iter @ scaled_constr_matrix @ col_scaling_iter
        scaled_Q_matrix = col_scaling_iter @ scaled_Q_matrix @ col_scaling_iter
        scaled_obj_vector = col_scaling_iter @ scaled_obj_vector
        
        # Compute cost scaling factor γ
        Q_col_norms = []
        for j in range(num_cols):
            col_data = np.abs(scaled_Q_matrix.getcol(j).data)
            if len(col_data) > 0:
                Q_col_norms.append(np.max(col_data))
        
        denominator = max(
            np.mean(Q_col_norms) if Q_col_norms else 1.0,
            np.max(np.abs(scaled_obj_vector)) if scaled_obj_vector.size > 0 else 1.0,
            1.0
        )
        obj_scaling_factor = 1.0 / denominator
        obj_scaling_factor = np.clip(obj_scaling_factor, ScalingLB, ScalingUB)
        
        # Apply cost scaling
        scaled_Q_matrix = obj_scaling_factor * scaled_Q_matrix
        scaled_obj_vector = obj_scaling_factor * scaled_obj_vector
        
        # Accumulate total scaling
        diagonal_scaling_total = diagonal_scaling_iter @ diagonal_scaling_total
        obj_scaling_factor_total *= obj_scaling_factor
        
        # Check convergence
        norm_constr_diff = scipy.sparse.linalg.norm(scaled_constr_matrix - previous_constr_matrix, ord='fro')
        norm_constr_prev = scipy.sparse.linalg.norm(previous_constr_matrix, ord='fro')
        norm_Q_diff = scipy.sparse.linalg.norm(scaled_Q_matrix - previous_Q_matrix, ord='fro')
        norm_Q_prev = scipy.sparse.linalg.norm(previous_Q_matrix, ord='fro')
        
        if norm_constr_prev > 0 and norm_Q_prev > 0:
            rel_constr_diff = norm_constr_diff / norm_constr_prev
            rel_Q_diff = norm_Q_diff / norm_Q_prev
            
            if rel_constr_diff < ScaleRelTol and rel_Q_diff < ScaleRelTol:
                break
    
    # Extract final column and row scaling
    col_scaling_total = scipy.sparse.diags(diagonal_scaling_total.diagonal()[:num_cols])
    row_scaling_total = scipy.sparse.diags(diagonal_scaling_total.diagonal()[num_cols:])
    
    return scaled_constr_matrix, scaled_Q_matrix, scaled_obj_vector, row_scaling_total, col_scaling_total
    

def scale_model(model: gp.Model, 
                method: str,
                ScalePasses: int = 5, 
                ScaleRelTol: float = 1e-4,
                ScalingLB: float = 1e-8, 
                ScalingUB: float = 1e8, 
                value_threshold: float = 1e-13,
                ScalingThreads: int = -1) -> ScaledModel:
    """
    Scale a Gurobi optimization model to improve numerical conditioning.
    
    Creates a scaled version of the input model using the specified scaling method.
    The scaled model can be solved, and the solution can be unscaled back to the
    original variable space.
    
    Parameters:
    -----------
    model : gp.Model
        The Gurobi model to scale
    method : str
        Scaling method to use. Options:
        - 'equilibration': Mean-based equilibration (works for LP, QP, QCP)
        - 'geometric_mean': Geometric mean scaling (LP only)
        - 'arithmetic_mean': Arithmetic mean scaling (LP only)
    ScalePasses : int, optional
        Maximum number of scaling iterations (default: 5)
    ScaleRelTol : float, optional
        Relative tolerance for convergence (default: 1e-4)
    ScalingLB : float, optional
        Lower bound for scaling factors to avoid extreme values (default: 1e-8)
    ScalingUB : float, optional
        Upper bound for scaling factors to avoid extreme values (default: 1e8)
    value_threshold : float, optional
        Threshold below which coefficients are set to zero (default: 1e-13)
    ScalingThreads : int, optional
        Number of parallel threads to use for scaling computations.
        -1 means use all available processors (default: -1)
        
    Returns:
    --------
    ScaledModel
        A scaled version of the input model with scaling information attached.
        Use getVarsUnscaled() to access unscaled solution values.
        
    Notes:
    ------
    - For models with quadratic objectives or constraints, only 'equilibration' method is supported
    - Integer and binary variables are not scaled (only continuous variables)
    - The scaled model includes scaling matrices stored as _col_scaling and _row_scaling attributes
    """
    # Get model data from original model
    model_data = ModelData.from_gurobi_model(model)
    
    # Compute list of columns to scale (don't scale binary/integer variables)
    cols_to_scale = []
    for i, var_type in enumerate(model_data.var_types):
        if var_type is not GRB.INTEGER or var_type is not GRB.BINARY:
            cols_to_scale.append(i)
    
    # Check if model has quadratic objective terms
    Q_matrix = model.getQ()
    
    if Q_matrix.nnz == 0: # No quadratic objective, use constraint matrix only
        # Compute scaled matrix and scaling factors
        if method == 'equilibration':
            scaled_matrix, row_scaling, col_scaling = equilibration(model_data.constr_matrix, cols_to_scale,
                                                                    ScalePasses, ScaleRelTol, n_jobs=ScalingThreads)
        elif method == 'geometric_mean':
            scaled_matrix, row_scaling, col_scaling = geometric_mean(model_data.constr_matrix, cols_to_scale,
                                                                    ScalePasses, ScaleRelTol, n_jobs=ScalingThreads)
        elif method == 'arithmetic_mean':
            scaled_matrix, row_scaling, col_scaling = arithmetic_mean(model_data.constr_matrix, cols_to_scale,
                                                                    ScalePasses, ScaleRelTol, n_jobs=ScalingThreads)
        # Scale objective vector (is done within quad_equilibration if Q present)
        obj_vector_scaled = col_scaling @ model_data.obj_vector
    else: # Consider Q in the scaling
        # Raise warning if other method than equilibration is selected
        if method != 'equilibration':
            warnings.warn("Equilibration is the only supported method for quadratic objectives. Using equilibration instead.", 
                        UserWarning)
        scaled_matrix, scaled_Q_matrix, obj_vector_scaled, row_scaling, col_scaling = quad_equilibration(model_data.constr_matrix, model_data.obj_vector, Q_matrix, cols_to_scale,
                                                                ScalePasses, ScaleRelTol,
                                                                ScalingLB=ScalingLB, ScalingUB=ScalingUB, n_jobs=ScalingThreads)
            
    # Compute scaled data
    rhs_vector_scaled = row_scaling @ model_data.rhs_vector
    col_diag = col_scaling.diagonal()
    # Avoid extreme inverse values
    col_diag_safe = np.clip(col_diag, ScalingLB, ScalingUB)
    col_scaling_inv = scipy.sparse.diags(1.0 / col_diag_safe)
    
    
    lb_vector_scaled = col_scaling_inv @ model_data.lb_vector
    ub_vector_scaled = col_scaling_inv @ model_data.ub_vector
    var_names_scaled = [name + "_scaled" for name in model_data.var_names]
    constr_names_scaled = [name + "_scaled" for name in model_data.constr_names]
    
    # Clean small coefficients
    scaled_matrix = threshold_small_coefficients(scaled_matrix, value_threshold)
    rhs_vector_scaled = threshold_small_coefficients(rhs_vector_scaled, value_threshold)
    obj_vector_scaled = threshold_small_coefficients(obj_vector_scaled, value_threshold)
    lb_vector_scaled = threshold_small_coefficients(lb_vector_scaled, value_threshold)
    ub_vector_scaled = threshold_small_coefficients(ub_vector_scaled, value_threshold)
    if Q_matrix.nnz > 0:
        scaled_Q_matrix = threshold_small_coefficients(scaled_Q_matrix, value_threshold)
   
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
        
    # Set objective sense
    model_scaled.ModelSense = model.ModelSense
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
    
    # Add quadratic objective term if present
    if Q_matrix.nnz > 0:
        lin_objective = model_scaled.getObjective()
        x_mvars = gp.MVar(model_scaled.getVars()) # Get matrix variable representation
        full_objective = lin_objective + x_mvars.T @ scaled_Q_matrix @ x_mvars
        model_scaled.setObjective(full_objective, model.ModelSense)
        model_scaled.update()
    # Scale quadratic constraints if present
    if model.isQCP:
        qconstrs = model.getQConstrs()
        # Process quadratic constraints in parallel
        qconstr_results = Parallel(n_jobs=ScalingThreads, prefer='threads')(
            delayed(_scale_single_qconstr)(qconstr, model, col_scaling)
            for qconstr in qconstrs
        )
        
        # Add scaled quadratic constraints to model
        quad_scaling_factors = []
        for Qc_scaled, q_scaled, sense, rhs_scaled, scaling_factor, name in qconstr_results:
            model_scaled.addMQConstr(
                Qc_scaled, q_scaled.toarray().flatten(), sense, rhs_scaled, name=name)
            quad_scaling_factors.append(scaling_factor)
        
        model_scaled.update()
        # Add scaling factors to scaled model
        model_scaled._quad_scaling_factors = quad_scaling_factors
    return model_scaled


def compute_constraint_violations(model: gp.Model, 
                                 unscaled_variables: Union[List[float], np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute constraint and bound violations for a given candidate solution.
    
    Evaluates how much a candidate solution violates the constraints and variable
    bounds of the model. This is useful for checking solution quality or
    assessing infeasibility.
    
    Parameters:
    -----------
    model : gp.Model
        The Gurobi model containing constraints
    unscaled_variables : list or np.ndarray
        Solution values in the same order as model.getVars()
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary with two keys:
        - 'constraints': dict mapping constraint names to violation values
        - 'bounds': dict mapping variable names to bound violation values
        
    Notes:
    ------
    Violation definitions:
        - For <= constraints: max(0, LHS - RHS)
        - For >= constraints: max(0, RHS - LHS)
        - For == constraints: |LHS - RHS|
        - For bounds: max(0, x - UB) + max(0, LB - x)
    """
    constraint_violations = {}
    bound_violations = {}
    vars_list = model.getVars()
    solution_dict = {var: val for var, val in zip(vars_list, unscaled_variables)}
    
    # Process bound violations
    for var, val in zip(vars_list, unscaled_variables):
        lb_vio = max(0.0, var.LB - val) if var.LB > -GRB.INFINITY else 0.0
        ub_vio = max(0.0, val - var.UB) if var.UB < GRB.INFINITY else 0.0
        bound_violations[var.VarName] = lb_vio + ub_vio # One of the two will be zero
    
    # Process linear constraints
    for constr in model.getConstrs():
        # Get constraint properties
        sense = constr.Sense
        rhs = constr.RHS
        name = constr.ConstrName
        
        # Compute LHS value
        row = model.getRow(constr)
        lhs_value = 0.0
        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            lhs_value += coeff * solution_dict.get(var, 0.0)
        
        # Compute violation based on sense
        if sense == GRB.LESS_EQUAL:  # <=
            violation = max(0.0, lhs_value - rhs)
        elif sense == GRB.GREATER_EQUAL:  # >=
            violation = max(0.0, rhs - lhs_value)
        elif sense == GRB.EQUAL:  # ==
            violation = abs(lhs_value - rhs)
        else:
            violation = 0.0  # Unknown sense
            
        constraint_violations[name] = violation
    
    # Process quadratic constraints
    try:
        import numpy as np
        
        for qconstr in model.getQConstrs():
            # Get constraint properties
            sense = qconstr.QCSense
            rhs = qconstr.QCRHS
            name = qconstr.QCName
            
            # Get quadratic and linear parts
            Q, c = model.getQCMatrices(qconstr)
            
            # Build solution vector in model variable order
            x = np.array([solution_dict.get(v, 0.0) for v in vars_list])
            
            # Compute LHS: x^T Q x + c^T x
            # Q is upper triangular, so we need to account for symmetry
            Q_full = Q + Q.T - np.diag(Q.diagonal())  # Make Q symmetric
            lhs_value = float(x.T @ Q_full @ x + c.T @ x)
            
            # Compute violation based on sense
            if sense == GRB.LESS_EQUAL:  # <=
                violation = max(0.0, lhs_value - rhs)
            elif sense == GRB.GREATER_EQUAL:  # >=
                violation = max(0.0, rhs - lhs_value)
            elif sense == GRB.EQUAL:  # ==
                violation = abs(lhs_value - rhs)
            else:
                violation = 0.0
                
            constraint_violations[name] = violation
            
    except ImportError:
        # numpy not available, skip quadratic constraints
        num_qconstrs = model.NumQConstrs
        if num_qconstrs > 0:
            print(f"Warning: {num_qconstrs} quadratic constraint(s) found but numpy is not available.")
            print("         Quadratic constraint violations will not be computed.")
    
    return {'constraints': constraint_violations, 'bounds': bound_violations}