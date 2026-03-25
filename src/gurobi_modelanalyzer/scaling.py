import gurobipy as gp
from gurobipy import GRB
import scipy.sparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union
import warnings
import time
import io
import logging
import sys


import gurobi_modelanalyzer.common as common

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False


def _capture_model_stats(model: gp.Model) -> str:
    """
    Capture the output of model.printStats() as a string.

    Parameters:
    -----------
    model : gp.Model
        The Gurobi model to analyze

    Returns:
    --------
    str
        The statistics output as a string
    """
    # Ensure model is updated before capturing stats
    model.update()

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        model.printStats()
        output = buffer.getvalue()
    finally:
        sys.stdout = old_stdout

    return output


def _extract_range_stats(stats: str) -> str:
    """
    Extract only range-related lines from model statistics.

    Parameters:
    -----------
    stats : str
        Full model statistics string from printStats()

    Returns:
    --------
    str
        Only the lines containing range information
    """
    range_lines = [line for line in stats.split(
        '\n') if 'range' in line.lower()]
    return '\n'.join(range_lines)


def _print_scaling_log(
        method: str,
        original_stats: str,
        scale_passes: int,
        iteration_logs: List[Dict],
        total_time: float,
        final_stats: str,
        scaling_time_limit: float = float('inf'),
        mode: str = 'final'):
    """
    Emit formatted scaling log entries via the module logger.

    Attach handlers to ``logging.getLogger('gurobi_modelanalyzer.scaling')``
    to control where output is written.  ``scale_model`` manages handler
    setup and teardown automatically based on its ``scaling_log`` and
    ``scaling_log_to_console`` parameters.

    Parameters:
    -----------
    method : str
        Scaling method name
    original_stats : str
        Statistics output from original model's printStats()
    scale_passes : int
        Number of scale passes requested
    iteration_logs : List[Dict]
        List of dictionaries containing iteration information
    total_time : float
        Total elapsed time for scaling
    final_stats : str
        Statistics output from scaled model's printStats()
    scaling_time_limit : float, optional
        Time limit for scaling (default: inf)
    mode : str, optional
        'header'    = emit header and iteration-table header
        'iteration' = emit a single iteration row
        'final'     = emit complete log as one block (default)
    """
    if mode == 'header':
        logger.info("\n" + "-" * 80)
        logger.info(f"Scaling Method: {method}")
        logger.info(f"Scale Passes:   {scale_passes}")
        if scaling_time_limit != float('inf'):
            logger.info(
                f"Time Limit:     {scaling_time_limit:.2f} seconds")
        logger.info("\nOriginal Model Statistics:")
        logger.info(original_stats.rstrip())
        logger.info("\n" + "-" * 80)
        logger.info(
            f"{'Scale Pass':<12} {'Rel. Change':<15} {'Time (s)':<15}")
        logger.info("-" * 80)
        return

    if mode == 'iteration':
        if len(iteration_logs) > 0:
            log = iteration_logs[-1]
            pass_num = log.get('pass', '-')
            rel_change = log.get('rel_change', 0.0)
            iter_time = log.get('time', 0.0)
            if isinstance(rel_change, float):
                logger.info(
                    f"{pass_num:<12} {rel_change:<15.6e} {iter_time:<15.6f}")
            else:
                logger.info(
                    f"{pass_num:<12} {rel_change:<15} {iter_time:<15.6f}")
        return

    # mode == 'final': emit complete log as a single block
    log_lines = []
    log_lines.append("\n" + "-" * 80)
    log_lines.append(f"Scaling Method: {method}")
    log_lines.append(f"Scale Passes:   {scale_passes}")
    if scaling_time_limit != float('inf'):
        log_lines.append(
            f"Time Limit:     {scaling_time_limit:.2f} seconds")
    log_lines.append("\nOriginal Model Statistics:")
    log_lines.append(original_stats.rstrip())
    log_lines.append("\n" + "-" * 80)
    log_lines.append(
        f"{'Scale Pass':<12} {'Rel. Change':<15} {'Time (s)':<15}")
    log_lines.append("-" * 80)
    for log in iteration_logs:
        pass_num = log.get('pass', '-')
        rel_change = log.get('rel_change', 0.0)
        iter_time = log.get('time', 0.0)
        if isinstance(rel_change, float):
            log_lines.append(
                f"{pass_num:<12} {rel_change:<15.6e} {iter_time:<15.6f}")
        else:
            log_lines.append(
                f"{pass_num:<12} {rel_change:<15} {iter_time:<15.6f}")
    log_lines.append("-" * 80)
    log_lines.append(f"\nScaling completed in {total_time:.6f} seconds")
    log_lines.append("\nScaled Model Ranges:")
    log_lines.append(_extract_range_stats(final_stats))
    log_lines.append("-" * 80 + "\n")
    logger.info("\n".join(log_lines))


class ScaledVar:
    """
    Wrapper around a Gurobi variable that provides access to unscaled
    values and bound violations.
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


class _ScaledConstraintBase:
    """
    Shared base for ScaledConstr and ScaledQConstr.
    Stores the wrapped Gurobi object, tracks unscaled violation, and
    forwards unknown attribute lookups to the wrapped object.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._unsc_violation = None

    @property
    def UnscViolation(self):
        """Unscaled constraint violation"""
        return self._unsc_violation

    @UnscViolation.setter
    def UnscViolation(self, value):
        self._unsc_violation = value

    def __getattr__(self, name):
        """Forward all other attributes to the underlying Gurobi object"""
        return getattr(self._wrapped, name)


class ScaledConstr(_ScaledConstraintBase):
    """
    Wrapper around a Gurobi constraint that provides access to
    unscaled violations.
    """

    def __init__(self, gurobi_constr):
        super().__init__(gurobi_constr)


class ScaledQConstr(_ScaledConstraintBase):
    """
    Wrapper around a Gurobi quadratic constraint that provides
    access to unscaled violations.
    """

    def __init__(self, gurobi_qconstr):
        super().__init__(gurobi_qconstr)


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

    def _get_cached_wrappers(self, cache_attr, get_method, wrapper_cls):
        """Lazy-initialise and return a list of constraint wrapper objects."""
        if getattr(self, cache_attr) is None:
            if self._original_model is None:
                raise ValueError(
                    "Original model not stored. "
                    "Cannot access constraint wrappers.")
            objects = get_method(self._original_model)
            setattr(self, cache_attr, [wrapper_cls(o) for o in objects])
        return getattr(self, cache_attr)

    def getConstrsUnscaled(self):
        """
        Get list of ScaledConstr objects with UnscViolation attributes.

        Returns:
        --------
        List[ScaledConstr]
            List of wrapped constraints with violation tracking
        """
        return self._get_cached_wrappers(
            '_scaled_constrs', lambda m: m.getConstrs(), ScaledConstr)

    def getQConstrsUnscaled(self):
        """
        Get list of ScaledQConstr objects with UnscViolation attributes.

        Returns:
        --------
        List[ScaledQConstr]
            List of wrapped quadratic constraints with violation tracking
        """
        return self._get_cached_wrappers(
            '_scaled_qconstrs', lambda m: m.getQConstrs(), ScaledQConstr)

    def ComputeUnscVio(self, original_model):
        """
        Compute unscaled constraint and bound violations using the
        unscaled variable values.
        Stores violations in constraint wrapper objects and variable
        wrapper objects,
        and tracks maximum violation.

        After calling this method, violations can be accessed via:
        - constraint.UnscViolation for each constraint
        - var.UnscBoundViolation for each variable's bound violation
        - model_scaled.MaxUnscVio for the maximum violation across
          all constraints and bounds
        - model_scaled.MaxUnscConstrVio for max constraint violation only
        - model_scaled.MaxUnscBoundVio for max bound violation only

        Parameters:
        -----------
        original_model : gp.Model
            The original unscaled model with same structure as this
            scaled model
        """
        # Store reference to original model
        self._original_model = original_model

        # Get unscaled solution values
        unscaled_vars = self.getVarsUnscaled()
        unscaled_values = [var.Xunsc for var in unscaled_vars]

        # Compute constraint and bound violations using the original model
        violations = compute_constraint_violations(
            original_model, unscaled_values)

        # Store violations in dictionaries
        self._constraint_violations = violations['constraints']
        self._bound_violations = violations['bounds']

        # Store violations in constraint wrappers
        scaled_constrs = self.getConstrsUnscaled()
        for scaled_constr in scaled_constrs:
            constr_name = scaled_constr.ConstrName
            scaled_constr.UnscViolation = self._constraint_violations.get(
                constr_name, 0.0)

        # Store violations in quadratic constraint wrappers
        if original_model.NumQConstrs > 0:
            scaled_qconstrs = self.getQConstrsUnscaled()
            for scaled_qconstr in scaled_qconstrs:
                qconstr_name = scaled_qconstr.QCName
                scaled_qconstr.UnscViolation = self._constraint_violations.get(
                    qconstr_name, 0.0)

        # Store bound violations in variable wrappers
        for i, var in enumerate(unscaled_vars):
            var_name = var.VarName.replace('_scaled', '')
            var.UnscBoundViolation = self._bound_violations.get(var_name, 0.0)

        # Compute and store maximum violations
        all_constraint_vios = list(self._constraint_violations.values())
        all_bound_vios = list(self._bound_violations.values())

        self._max_unsc_constr_vio = max(
            all_constraint_vios) if all_constraint_vios else 0.0
        self._max_unsc_bound_vio = max(
            all_bound_vios) if all_bound_vios else 0.0
        self._max_unsc_vio = max(
            self._max_unsc_constr_vio,
            self._max_unsc_bound_vio)

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
        Get the maximum unscaled constraint violation (linear and
        quadratic constraints only).

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

    @property
    def ScalingTime(self):
        """
        Get the time taken to scale the model.

        Returns:
        --------
        float
            Scaling time in seconds, or None if not available
        """
        return getattr(self, '_scaling_time', None)

    @property
    def ColScaling(self):
        """
        Get the column scaling matrix.

        Returns:
        --------
        scipy.sparse.csr_matrix
            Diagonal matrix with column scaling factors, or None if
            not available
        """
        return getattr(self, '_col_scaling', None)

    @property
    def RowScaling(self):
        """
        Get the row scaling matrix.

        Returns:
        --------
        scipy.sparse.csr_matrix
            Diagonal matrix with row scaling factors, or None if not available
        """
        return getattr(self, '_row_scaling', None)

    def ComputeUnscObj(self, original_model):
        """
        Compute the unscaled objective value using original model coefficients
        and unscaled variable values.

        This method should be called after optimization to get the objective
        value in the original (unscaled) space.

        Parameters:
        -----------
        original_model : gp.Model
            The original unscaled model with same structure as this
            scaled model
        """
        # Store reference to original model if not already set
        self._original_model = original_model

        # Get unscaled solution values
        unscaled_vars = self.getVarsUnscaled()
        unscaled_values = np.array([var.Xunsc for var in unscaled_vars])

        # Get original objective coefficients
        orig_obj = np.array(original_model.getAttr("Obj"))

        # Compute linear objective contribution
        linear_obj = np.dot(orig_obj, unscaled_values)

        # Check for quadratic objective
        q_matrix = original_model.getQ()
        if q_matrix.nnz > 0:
            # Quadratic contribution: x^T q x
            # q is upper triangular, need full symmetric form
            q_full = q_matrix + q_matrix.T - \
                scipy.sparse.diags(q_matrix.diagonal())
            quad_obj = float(unscaled_values @ q_full @ unscaled_values)
        else:
            quad_obj = 0.0

        self._unsc_obj_val = linear_obj + quad_obj

    @property
    def UnscObjVal(self):
        """
        Get the unscaled objective value.

        This is the objective value computed using original model coefficients
        and unscaled variable values.

        Returns:
        --------
        float
            Unscaled objective value, or None if not computed.
            Call ComputeUnscObj(original_model) first.
        """
        return getattr(self, '_unsc_obj_val', None)


@dataclass
class ModelData:
    """
    Encapsulates Gurobi model data extracted prior to scaling.
    """
    constr_matrix: scipy.sparse.csr_matrix = field(default=None)
    rhs_vector: np.ndarray = field(default=None)
    constr_sense: List[str] = field(default=None)
    obj_vector: np.ndarray = field(default=None)
    ub_vector: np.ndarray = field(default=None)
    lb_vector: np.ndarray = field(default=None)
    var_types: List[str] = field(default=None)
    var_names: List[str] = field(default=None)
    constr_names: List[str] = field(default=None)

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


def _row_scale_factor(row_data: np.ndarray, method: str) -> float:
    """Compute a row scaling factor for the given method."""
    if method in ('equilibration', 'arithmetic_mean'):
        return 1.0 / np.mean(row_data)
    else:  # geometric_mean
        return 1.0 / np.sqrt(np.min(row_data) * np.max(row_data))


def _col_scale_factor(col_data: np.ndarray, method: str) -> float:
    """Compute a column scaling factor for the given method."""
    if method == 'equilibration':
        return 1.0 / np.max(col_data)
    elif method == 'arithmetic_mean':
        return 1.0 / np.mean(col_data)
    else:  # geometric_mean
        return 1.0 / np.sqrt(np.min(col_data) * np.max(col_data))


def _iterative_scaling(
        constr_matrix: scipy.sparse.csr_matrix,
        cols_to_scale: List[int],
        rows_to_scale: List[int],
        scale_passes: int,
        scale_rel_tol: float,
        method: str,
        scaling_time_limit: float = float('inf'),
) -> Tuple[
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        List[Dict]]:
    """
    Shared iterative row/column scaling loop used by equilibration,
    geometric_mean, and arithmetic_mean.

    Parameters:
    -----------
    constr_matrix : scipy.sparse.csr_matrix
        The constraint matrix to scale
    cols_to_scale : List[int]
        Column indices to scale
    rows_to_scale : List[int]
        Row indices to scale
    scale_passes : int
        Maximum number of scaling iterations
    scale_rel_tol : float
        Relative tolerance for convergence check
    method : str
        Scaling method ('equilibration', 'geometric_mean', or
        'arithmetic_mean')
    scaling_time_limit : float, optional
        Time limit in seconds (default: inf - no limit)

    Returns:
    --------
    Tuple[
        scipy.sparse.csr_matrix, scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix, List[Dict]]
        - scaled_constr_matrix: The scaled constraint matrix
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
        - iteration_logs: List of iteration information dictionaries
    """
    scaled_constr_matrix = constr_matrix
    num_rows, num_cols = constr_matrix.shape
    row_scaling_total = scipy.sparse.eye(num_rows, format='csr')
    col_scaling_total = scipy.sparse.eye(num_cols, format='csr')
    iteration_logs = []
    total_elapsed_time = 0.0

    rows_to_scale_set = set(rows_to_scale)
    cols_to_scale_set = set(cols_to_scale)

    scaled_constr_matrix_csc = scaled_constr_matrix.tocsc()

    for completed_scale_passes in range(scale_passes):
        iter_start_time = time.time()
        scaled_constr_matrix = scaled_constr_matrix_csc.tocsr()

        # Compute row scaling factors (skip excluded rows)
        row_factors = np.ones(num_rows)
        for i in rows_to_scale_set:
            row_data = np.abs(scaled_constr_matrix.getrow(i).data)
            if len(row_data) > 0:
                row_factors[i] = _row_scale_factor(row_data, method)

        row_scaling_iter = scipy.sparse.diags(row_factors)
        scaled_constr_matrix = row_scaling_iter @ scaled_constr_matrix
        row_scaling_total = row_scaling_iter @ row_scaling_total

        scaled_constr_matrix_csc = scaled_constr_matrix.tocsc()

        # Compute column scaling factors via direct CSC array access
        col_factors_full = np.ones(num_cols)
        csc_data = np.abs(scaled_constr_matrix_csc.data)
        csc_indptr = scaled_constr_matrix_csc.indptr
        for j in range(num_cols):
            if j in cols_to_scale_set:
                start_idx = csc_indptr[j]
                end_idx = csc_indptr[j + 1]
                if end_idx > start_idx:
                    col_factors_full[j] = _col_scale_factor(
                        csc_data[start_idx:end_idx], method)

        col_scaling_iter = scipy.sparse.diags(col_factors_full)
        scaled_constr_matrix_csc = scaled_constr_matrix_csc @ col_scaling_iter
        col_scaling_total = col_scaling_total @ col_scaling_iter

        rel_change = max(
            np.max(np.abs(row_factors - 1.0)),
            np.max(np.abs(col_factors_full - 1.0)))

        iter_time = time.time() - iter_start_time
        total_elapsed_time += iter_time
        iteration_logs.append({
            'pass': completed_scale_passes + 1,
            'rel_change': rel_change,
            'time': iter_time
        })
        _print_scaling_log(
            '', '', 0, iteration_logs, 0.0, '',
            mode='iteration')

        if rel_change < scale_rel_tol:
            break
        if total_elapsed_time >= scaling_time_limit:
            break

    scaled_constr_matrix = scaled_constr_matrix_csc.tocsr()
    return (
        scaled_constr_matrix, row_scaling_total,
        col_scaling_total, iteration_logs
    )


def _scale_single_qconstr(
        qconstr: gp.QConstr,
        model: gp.Model,
        col_scaling: scipy.sparse.csr_matrix,
        skip_row_scale: bool = False,
) -> Tuple[
        scipy.sparse.csr_matrix, np.ndarray,
        str, float, float, str]:
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
    skip_row_scale : bool, optional
        If True, skip the row-level normalisation (scaling_factor=1).
        Column scaling is still applied. Use when constr._scale=0
        (default: False)

    Returns:
    --------
    Tuple containing:
        - qc_scaled: Scaled quadratic matrix
        - q_scaled: Scaled linear vector (as flattened numpy array)
        - sense: Constraint sense
        - rhs_scaled: Scaled RHS
        - scaling_factor: Computed scaling factor
        - name: Constraint name
    """
    # Extract data from constraint
    qc, q = model.getQCMatrices(qconstr)
    qc = col_scaling @ qc @ col_scaling
    q = col_scaling @ q
    # Convert q to dense array and flatten (getQCMatrices returns sparse
    # column matrix)
    if scipy.sparse.issparse(q):
        q = np.asarray(q.todense()).flatten()
    else:
        q = np.asarray(q).flatten()
    rhs = qconstr.QCRHS
    sense = qconstr.QCSense
    name = qconstr.QCName + "_scaled"

    # Compute Frobenius norm efficiently for upper triangular matrix
    # For symmetric matrix: ||A||_F^2 = sum(A_ij^2) for all i,j
    # For upper triangular q: ||q + q^T - diag(q)||_F^2 =
    # 2*sum(Q_ij^2) - sum(Q_ii^2)
    # Faster: sqrt(2 * ||q||_F^2 - ||diag(q)||_2^2)
    if qc.nnz > 0:
        qc_norm_sq = np.sum(qc.data ** 2)  # ||q||_F^2
        qc_diag_norm_sq = np.sum(qc.diagonal() ** 2)  # ||diag(q)||_2^2
        qc_full_norm = np.sqrt(2.0 * qc_norm_sq - qc_diag_norm_sq)
    else:
        qc_full_norm = 0.0

    # Compute scaling factor for constraint
    q_norm = np.linalg.norm(q) if q.size > 0 else 0.0
    if skip_row_scale:
        scaling_factor = 1.0
    else:
        scaling_factor = 1.0 / max(qc_full_norm, q_norm, abs(rhs), 1.0)

    qc_scaled = scaling_factor * qc
    q_scaled = scaling_factor * q
    rhs_scaled = scaling_factor * rhs

    return qc_scaled, q_scaled, sense, rhs_scaled, scaling_factor, name


def threshold_small_coefficients(
        data: Union[scipy.sparse.spmatrix, np.ndarray],
        value_threshold: float = 1e-13,
) -> Union[scipy.sparse.csr_matrix, np.ndarray]:
    """
    Set coefficients below threshold to zero.
    Works for both sparse matrices and numpy arrays.

    Parameters:
    -----------
    data : scipy.sparse matrix or np.ndarray
        The data to threshold
    value_threshold : float, optional
        Absolute value threshold below which coefficients are set to
        zero (default: 1e-13)

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


def equilibration(
        constr_matrix: scipy.sparse.csr_matrix,
        cols_to_scale: List[int],
        rows_to_scale: List[int],
        scale_passes: int,
        scale_rel_tol: float,
        scaling_time_limit: float = float('inf'),
) -> Tuple[
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        List[Dict]]:
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
    rows_to_scale : List[int]
        List of row indices to scale (constraints not in this list
        keep a row scaling factor of 1)
    scale_passes : int
        Maximum number of scaling iterations
    scale_rel_tol : float
        Relative tolerance for convergence check
    scaling_time_limit : float, optional
        Time limit in seconds for scaling iterations (default: inf - no limit)

    Returns:
    --------
    Tuple[
        scipy.sparse.csr_matrix, scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix, List[Dict]]
        - scaled_constr_matrix: The scaled constraint matrix
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
        - iteration_logs: List of iteration information dictionaries
    """
    return _iterative_scaling(
        constr_matrix, cols_to_scale, rows_to_scale,
        scale_passes, scale_rel_tol, 'equilibration',
        scaling_time_limit)


def geometric_mean(
        constr_matrix: scipy.sparse.csr_matrix,
        cols_to_scale: List[int],
        rows_to_scale: List[int],
        scale_passes: int,
        scale_rel_tol: float,
        scaling_time_limit: float = float('inf'),
) -> Tuple[
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        List[Dict]]:
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
    rows_to_scale : List[int]
        List of row indices to scale (constraints not in this list
        keep a row scaling factor of 1)
    scale_passes : int
        Maximum number of scaling iterations
    scale_rel_tol : float
        Relative tolerance for convergence check
    scaling_time_limit : float, optional
        Time limit in seconds for scaling iterations (default: inf - no limit)

    Returns:
    --------
    Tuple[
        scipy.sparse.csr_matrix, scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix, List[Dict]]
        - scaled_constr_matrix: The scaled constraint matrix
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
        - iteration_logs: List of iteration information dictionaries
    """
    return _iterative_scaling(
        constr_matrix, cols_to_scale, rows_to_scale,
        scale_passes, scale_rel_tol, 'geometric_mean',
        scaling_time_limit)


def arithmetic_mean(
        constr_matrix: scipy.sparse.csr_matrix,
        cols_to_scale: List[int],
        rows_to_scale: List[int],
        scale_passes: int,
        scale_rel_tol: float,
        scaling_time_limit: float = float('inf'),
) -> Tuple[
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        List[Dict]]:
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
    rows_to_scale : List[int]
        List of row indices to scale (constraints not in this list
        keep a row scaling factor of 1)
    scale_passes : int
        Maximum number of scaling iterations
    scale_rel_tol : float
        Relative tolerance for convergence check
    scaling_time_limit : float, optional
        Time limit in seconds for scaling iterations (default: inf - no limit)

    Returns:
    --------
    Tuple[
        scipy.sparse.csr_matrix, scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix, List[Dict]]
        - scaled_constr_matrix: The scaled constraint matrix
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
        - iteration_logs: List of iteration information dictionaries
    """
    return _iterative_scaling(
        constr_matrix, cols_to_scale, rows_to_scale,
        scale_passes, scale_rel_tol, 'arithmetic_mean',
        scaling_time_limit)


def quad_equilibration(
        constr_matrix: scipy.sparse.csr_matrix,
        obj_vector: np.ndarray,
        q_matrix: scipy.sparse.coo_matrix,
        cols_to_scale: List[int],
        rows_to_scale: List[int],
        scale_passes: int,
        scale_rel_tol: float,
        scaling_lb: float = 1e-8,
        scaling_ub: float = 1e8,
        scaling_time_limit: float = float('inf'),
) -> Tuple[
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        np.ndarray,
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        List[Dict]]:
    """
    Scale quadratic program using KKT-based equilibration method.

    Scales the constraint matrix, quadratic objective matrix (q), and linear
    objective vector jointly by building a KKT matrix and
    applying equilibration
    to both the variable/constraint scaling and the objective scaling.

    Parameters:
    -----------
    constr_matrix : scipy.sparse.csr_matrix
        The constraint matrix to scale
    obj_vector : np.ndarray
        Linear objective coefficient vector
    q_matrix : scipy.sparse.coo_matrix
        Quadratic objective matrix (Hessian)
    cols_to_scale : List[int]
        List of column indices to scale (typically continuous variables)
    rows_to_scale : List[int]
        List of row indices to scale (constraints not in this list
        keep a row scaling factor of 1)
    scale_passes : int
        Maximum number of scaling iterations
    scale_rel_tol : float
        Relative tolerance for convergence check
    scaling_lb : float, optional
        Lower bound for scaling factors (default: 1e-8)
    scaling_ub : float, optional
        Upper bound for scaling factors (default: 1e8)
    scaling_time_limit : float, optional
        Time limit in seconds for scaling iterations (default: inf - no limit)

    Returns:
    --------
    Tuple[
        scipy.sparse.csr_matrix, scipy.sparse.csr_matrix,
        np.ndarray, scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix, List[Dict]]
        - scaled_constr_matrix: The scaled constraint matrix
        - scaled_q_matrix: The scaled quadratic objective matrix
        - scaled_obj_vector: The scaled linear objective vector
        - row_scaling_total: Cumulative row scaling matrix (diagonal)
        - col_scaling_total: Cumulative column scaling matrix (diagonal)
        - iteration_logs: List of iteration information dictionaries
    """
    scaled_constr_matrix = constr_matrix.copy()
    scaled_q_matrix = q_matrix.copy()
    scaled_obj_vector = obj_vector.copy()

    # Initialize scaling matrices
    num_rows, num_cols = constr_matrix.shape
    diagonal_scaling_total = scipy.sparse.eye(
        num_rows + num_cols, format='csr')
    obj_scaling_factor_total = 1.0
    zero_block = scipy.sparse.csr_matrix((num_rows, num_rows))
    iteration_logs = []
    total_elapsed_time = 0.0

    for completed_scale_passes in range(scale_passes):
        iter_start_time = time.time()
        previous_constr_matrix = scaled_constr_matrix.copy()
        previous_q_matrix = scaled_q_matrix.copy()

        # Build KKT matrix from CURRENT scaled matrices and convert to CSC for
        # column operations
        kkt_matrix = scipy.sparse.bmat([
            [scaled_q_matrix, scaled_constr_matrix.T],
            [scaled_constr_matrix, zero_block]
        ]).tocsc()  # CSC format for efficient column access

        # Compute diagonal scaling factors using CSC direct access (much faster
        # than getcol)
        diagonal_factors = np.ones(num_rows + num_cols)
        cols_to_scale_set = set(cols_to_scale)
        rows_to_scale_set = set(rows_to_scale)

        # Access CSC internal arrays directly for speed
        kkt_data = np.abs(kkt_matrix.data)
        kkt_indptr = kkt_matrix.indptr

        for i in range(num_rows + num_cols):
            # Skip integer/binary variable columns
            if i < num_cols and i not in cols_to_scale_set:
                continue
            # Skip excluded constraint rows
            if i >= num_cols and (i - num_cols) not in rows_to_scale_set:
                continue

            # Get column data using direct array access
            start_idx = kkt_indptr[i]
            end_idx = kkt_indptr[i + 1]

            if end_idx > start_idx:  # Column has data
                col_data = kkt_data[start_idx:end_idx]
                max_val = np.max(col_data)
                scaling_factor = 1.0 / np.sqrt(max_val)
                diagonal_factors[i] = np.clip(
                    scaling_factor, scaling_lb, scaling_ub)
        diagonal_scaling_iter = scipy.sparse.diags(diagonal_factors)

        # Extract column and row scaling from THIS iteration
        col_scaling_iter = scipy.sparse.diags(
            diagonal_scaling_iter.diagonal()[:num_cols])
        row_scaling_iter = scipy.sparse.diags(
            diagonal_scaling_iter.diagonal()[num_cols:])

        # Apply M equilibration scaling
        scaled_constr_matrix = (
            row_scaling_iter @ scaled_constr_matrix @ col_scaling_iter
        )
        scaled_q_matrix = col_scaling_iter @ scaled_q_matrix @ col_scaling_iter
        scaled_obj_vector = col_scaling_iter @ scaled_obj_vector

        # Compute cost scaling factor γ
        q_col_norms = []
        for j in range(num_cols):
            col_data = np.abs(scaled_q_matrix.getcol(j).data)
            if len(col_data) > 0:
                q_col_norms.append(np.max(col_data))

        denominator = max(
            np.mean(q_col_norms) if q_col_norms else 1.0,
            np.max(np.abs(scaled_obj_vector))
            if scaled_obj_vector.size > 0 else 1.0,
            1.0,
        )
        obj_scaling_factor = 1.0 / denominator
        obj_scaling_factor = np.clip(
            obj_scaling_factor, scaling_lb, scaling_ub)

        # Apply cost scaling
        scaled_q_matrix = obj_scaling_factor * scaled_q_matrix
        scaled_obj_vector = obj_scaling_factor * scaled_obj_vector

        # Accumulate total scaling
        diagonal_scaling_total = diagonal_scaling_iter @ diagonal_scaling_total
        obj_scaling_factor_total *= obj_scaling_factor

        # Check convergence
        norm_constr_diff = scipy.sparse.linalg.norm(
            scaled_constr_matrix - previous_constr_matrix, ord='fro')
        norm_constr_prev = scipy.sparse.linalg.norm(
            previous_constr_matrix, ord='fro')
        norm_q_diff = scipy.sparse.linalg.norm(
            scaled_q_matrix - previous_q_matrix, ord='fro')
        norm_q_prev = scipy.sparse.linalg.norm(previous_q_matrix, ord='fro')

        rel_change = 0.0
        if norm_constr_prev > 0 and norm_q_prev > 0:
            rel_constr_diff = norm_constr_diff / norm_constr_prev
            rel_q_diff = norm_q_diff / norm_q_prev
            rel_change = max(rel_constr_diff, rel_q_diff)

        iter_time = time.time() - iter_start_time
        total_elapsed_time += iter_time
        iteration_logs.append({
            'pass': completed_scale_passes + 1,
            'rel_change': rel_change,
            'time': iter_time
        })

        # Emit iteration progress
        _print_scaling_log(
            '',
            '',
            0,
            iteration_logs,
            0.0,
            '',
            mode='iteration')

        if norm_constr_prev > 0 and norm_q_prev > 0:
            rel_constr_diff = norm_constr_diff / norm_constr_prev
            rel_q_diff = norm_q_diff / norm_q_prev

            if rel_constr_diff < scale_rel_tol and rel_q_diff < scale_rel_tol:
                break

        # Check time limit
        if total_elapsed_time >= scaling_time_limit:
            break

    # Extract final column and row scaling
    col_scaling_total = scipy.sparse.diags(
        diagonal_scaling_total.diagonal()[:num_cols])
    row_scaling_total = scipy.sparse.diags(
        diagonal_scaling_total.diagonal()[num_cols:])

    return (
        scaled_constr_matrix, scaled_q_matrix,
        scaled_obj_vector, row_scaling_total,
        col_scaling_total, iteration_logs
    )


def scale_model(model: gp.Model,
                method: str,
                scale_passes: int = 5,
                scale_rel_tol: float = 1e-4,
                scaling_lb: float = 1e-8,
                scaling_ub: float = 1e8,
                value_threshold: float = 1e-13,
                scaling_time_limit: float = float('inf'),
                scaling_log: str = "",
                scaling_log_to_console: int = 1) -> ScaledModel:
    """
    Scale a Gurobi optimization model to improve numerical conditioning.

    Creates a scaled version of the input model using the specified
    scaling method. The scaled model can be solved, and the solution
    can be unscaled back to the
    original variable space.

    Parameters:
    -----------
    model : gp.Model
        The Gurobi model to scale
    method : str
        Scaling method to use. Options:
        - 'equilibration': Mean-based equilibration (works for LP, QP, QCP)
        - 'geometric_mean': Geometric mean scaling (LP, QCP; not QP)
        - 'arithmetic_mean': Arithmetic mean scaling (LP, QCP; not QP)
    scale_passes : int, optional
        Maximum number of scaling iterations (default: 5)
    scale_rel_tol : float, optional
        Relative tolerance for convergence (default: 1e-4)
    scaling_lb : float, optional
        Lower bound for scaling factors to avoid extreme values (default: 1e-8)
    scaling_ub : float, optional
        Upper bound for scaling factors to avoid extreme values (default: 1e8)
    value_threshold : float, optional
        Threshold below which coefficients are set to zero (default: 1e-13)
    scaling_time_limit : float, optional
        Time limit in seconds for scaling iterations. Scaling will
        stop when this limit is reached and use the latest scaled
        matrices (default: inf - no limit)
    scaling_log : str, optional
        File path to write scaling log to. Empty string means no
        file output (default: "")
    scaling_log_to_console : int, optional
        1 to print scaling log to console, 0 to suppress console
        output (default: 1)

    Returns:
    --------
    ScaledModel
        A scaled version of the input model with scaling information attached.
        Use getVarsUnscaled() to access unscaled solution values.

    Notes:
    ------
    - For models with quadratic objectives, only 'equilibration'
      method is supported
    - For models with quadratic constraints (but no quadratic
      objective), all methods work
    - Integer and binary variables are never scaled regardless of any
      _scale attribute
    - Set var._scale = 0 on a continuous variable to exclude it from
      column scaling
    - Set constr._scale = 0 on a linear or quadratic constraint to
      exclude it from row scaling
    - The scaled model includes scaling matrices stored as
      _col_scaling and _row_scaling attributes
    """
    # Start timing
    total_start_time = time.time()

    # Ensure model is updated
    model.update()

    # Manage OutputFlag: temporarily enable if needed for logging
    original_output_flag = model.Params.OutputFlag
    needs_output = scaling_log_to_console or scaling_log
    if original_output_flag == 0 and needs_output:
        model.setParam('OutputFlag', 1)

    # Configure logging handlers for this scaling call
    _log_handlers: List[logging.Handler] = []
    if scaling_log_to_console:
        _console_handler = logging.StreamHandler()
        _console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(_console_handler)
        _log_handlers.append(_console_handler)
    if scaling_log:
        _file_handler = logging.FileHandler(scaling_log, mode='w')
        _file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(_file_handler)
        _log_handlers.append(_file_handler)

    # Get model data from original model
    model_data = ModelData.from_gurobi_model(model)

    # Capture original model statistics
    original_stats = _capture_model_stats(model)

    # Compute columns to scale: skip integer/binary unconditionally;
    # skip continuous variables explicitly marked with _scale=0.
    gurobi_vars = model.getVars()
    cols_to_scale = []
    for i, var_type in enumerate(model_data.var_types):
        if var_type in (GRB.INTEGER, GRB.BINARY):
            continue
        if getattr(gurobi_vars[i], '_scale', 1) == 0:
            continue
        cols_to_scale.append(i)

    # Compute rows to scale: skip any constraint marked with _scale=0.
    rows_to_scale = [
        i for i, constr in enumerate(model.getConstrs())
        if getattr(constr, '_scale', 1) != 0
    ]

    # Check if model has quadratic objective terms
    q_matrix = model.getQ()

    iteration_logs = []

    # Print log header if logging is enabled
    _print_scaling_log(
        method,
        original_stats,
        scale_passes,
        [],
        0.0,
        "",
        scaling_time_limit,
        mode='header')

    if q_matrix.nnz == 0:  # No quadratic objective, use constraint matrix only
        # Compute scaled matrix and scaling factors
        if method == 'equilibration':
            (scaled_matrix, row_scaling, col_scaling,
             iteration_logs) = equilibration(
                model_data.constr_matrix, cols_to_scale,
                rows_to_scale, scale_passes, scale_rel_tol,
                scaling_time_limit=scaling_time_limit)
        elif method == 'geometric_mean':
            (scaled_matrix, row_scaling, col_scaling,
             iteration_logs) = geometric_mean(
                model_data.constr_matrix, cols_to_scale,
                rows_to_scale, scale_passes, scale_rel_tol,
                scaling_time_limit=scaling_time_limit)
        elif method == 'arithmetic_mean':
            (scaled_matrix, row_scaling, col_scaling,
             iteration_logs) = arithmetic_mean(
                model_data.constr_matrix, cols_to_scale,
                rows_to_scale, scale_passes, scale_rel_tol,
                scaling_time_limit=scaling_time_limit)
        # Scale objective vector (is done within quad_equilibration if q
        # present)
        obj_vector_scaled = col_scaling @ model_data.obj_vector
    else:  # Consider q in the scaling
        # Raise warning if other method than equilibration is selected
        if method != 'equilibration':
            warnings.warn(
                "Equilibration is the only supported method for "
                "quadratic objectives. Using equilibration instead.",
                UserWarning)
        (scaled_matrix, scaled_q_matrix, obj_vector_scaled,
         row_scaling, col_scaling,
         iteration_logs) = quad_equilibration(
            model_data.constr_matrix, model_data.obj_vector,
            q_matrix, cols_to_scale, rows_to_scale,
            scale_passes, scale_rel_tol,
            scaling_lb=scaling_lb, scaling_ub=scaling_ub,
            scaling_time_limit=scaling_time_limit)

    # Print separator before model building phase
    logger.info("-" * 80)
    logger.info("Building scaled model...")

    # Compute scaled data
    rhs_vector_scaled = row_scaling @ model_data.rhs_vector
    col_diag = col_scaling.diagonal()
    # Avoid extreme inverse values
    col_diag_safe = np.clip(col_diag, scaling_lb, scaling_ub)
    col_scaling_inv = scipy.sparse.diags(1.0 / col_diag_safe)

    lb_vector_scaled = col_scaling_inv @ model_data.lb_vector
    ub_vector_scaled = col_scaling_inv @ model_data.ub_vector
    var_names_scaled = [name + "_scaled" for name in model_data.var_names]
    constr_names_scaled = [
        name + "_scaled" for name in model_data.constr_names]

    # Clean small coefficients
    scaled_matrix = threshold_small_coefficients(
        scaled_matrix, value_threshold)
    rhs_vector_scaled = threshold_small_coefficients(
        rhs_vector_scaled, value_threshold)
    obj_vector_scaled = threshold_small_coefficients(
        obj_vector_scaled, value_threshold)
    lb_vector_scaled = threshold_small_coefficients(
        lb_vector_scaled, value_threshold)
    ub_vector_scaled = threshold_small_coefficients(
        ub_vector_scaled, value_threshold)
    if q_matrix.nnz > 0:
        scaled_q_matrix = threshold_small_coefficients(
            scaled_q_matrix, value_threshold)

    # Create linear terms of ScaledModel with scaled data using matrix API
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
    if q_matrix.nnz > 0:
        lin_objective = model_scaled.getObjective()
        # Get matrix variable representation
        x_mvars = gp.MVar(model_scaled.getVars())
        full_objective = lin_objective + x_mvars.T @ scaled_q_matrix @ x_mvars
        model_scaled.setObjective(full_objective, model.ModelSense)
        model_scaled.update()
    # Scale quadratic constraints if present
    if model.isQCP:
        logger.info("Scaling quadratic constraints...")
        qconstrs = model.getQConstrs()
        # Process quadratic constraints
        qconstr_results = [
            _scale_single_qconstr(
                qconstr, model, col_scaling,
                skip_row_scale=(getattr(qconstr, '_scale', 1) == 0))
            for qconstr in qconstrs
        ]

        # Add scaled quadratic constraints to model
        quad_scaling_factors = []
        for (qc_scaled, q_scaled, sense,
             rhs_scaled, scaling_factor, name) in qconstr_results:
            model_scaled.addMQConstr(
                qc_scaled, q_scaled, sense, rhs_scaled, name=name)
            quad_scaling_factors.append(scaling_factor)

        model_scaled.update()
        # Add scaling factors to scaled model
        model_scaled._quad_scaling_factors = quad_scaling_factors

    # Compute total time and capture final statistics
    total_time = time.time() - total_start_time
    final_stats = _capture_model_stats(model_scaled)

    # Store scaling time as model attribute
    model_scaled._scaling_time = total_time

    # Emit scaling footer with final stats
    logger.info("-" * 80)
    logger.info(f"\nScaling completed in {total_time:.6f} seconds")
    logger.info("\nScaled Model Ranges:")
    logger.info(_extract_range_stats(final_stats))
    logger.info("-" * 80 + "\n")

    # Remove logging handlers added for this call
    for _h in _log_handlers:
        _h.close()
        logger.removeHandler(_h)

    # Restore original OutputFlag if we changed it
    if original_output_flag == 0 and needs_output:
        model.setParam('OutputFlag', 0)

    return model_scaled


def _compute_violation(sense: str, lhs_value: float, rhs: float) -> float:
    """Compute constraint violation given sense, LHS value, and RHS."""
    if sense == GRB.LESS_EQUAL:
        return max(0.0, lhs_value - rhs)
    elif sense == GRB.GREATER_EQUAL:
        return max(0.0, rhs - lhs_value)
    elif sense == GRB.EQUAL:
        return abs(lhs_value - rhs)
    return 0.0


def compute_constraint_violations(
        model: gp.Model,
        unscaled_variables: Union[List[float], np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute constraint and bound violations for a given candidate solution.

    Evaluates how much a candidate solution violates the constraints
    and variable
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
    solution_dict = {
        var: val for var,
        val in zip(
            vars_list,
            unscaled_variables)}

    # Process bound violations
    for var, val in zip(vars_list, unscaled_variables):
        lb_vio = max(0.0, var.LB - val) if var.LB > -GRB.INFINITY else 0.0
        ub_vio = max(0.0, val - var.UB) if var.UB < GRB.INFINITY else 0.0
        bound_violations[var.VarName] = lb_vio + \
            ub_vio  # One of the two will be zero

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

        constraint_violations[name] = _compute_violation(
            sense, lhs_value, rhs)

    # Process quadratic constraints
    try:
        for qconstr in model.getQConstrs():
            # Get constraint properties
            sense = qconstr.QCSense
            rhs = qconstr.QCRHS
            name = qconstr.QCName

            # Get quadratic and linear parts
            q, c = model.getQCMatrices(qconstr)

            # Build solution vector in model variable order
            x = np.array([solution_dict.get(v, 0.0) for v in vars_list])

            # Compute LHS: x^T q x + c^T x
            # q is upper triangular, so we need to account for symmetry
            q_full = q + q.T - np.diag(q.diagonal())  # Make q symmetric
            lhs_value = float(x.T @ q_full @ x + c.T @ x)

            constraint_violations[name] = _compute_violation(
                sense, lhs_value, rhs)

    except ImportError:
        # numpy not available, skip quadratic constraints
        num_qconstrs = model.NumQConstrs
        if num_qconstrs > 0:
            print(
                f"Warning: {num_qconstrs} quadratic constraint(s) "
                f"found but numpy is not available.")
            print(
                "         Quadratic constraint violations "
                "will not be computed.")

    return {'constraints': constraint_violations, 'bounds': bound_violations}
