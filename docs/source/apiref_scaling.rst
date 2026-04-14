.. _ScalingAPIRefLabel:

API Reference
#############


.. _APIscale_modelLabel:

.. py:function:: gurobi_modelanalyzer.scale_model(model, method, scale_passes=5, scale_rel_tol=1e-4, scaling_lb=1e-8, scaling_ub=1e8, value_threshold=1e-13, scaling_time_limit=inf, scaling_log="", scaling_log_to_console=1, init_scaling=0, env=None)

   Scale a Gurobi optimization model to improve numerical conditioning.

   Creates a scaled copy of the input model using the specified scaling
   method. The scaled model can be optimized directly and provides methods
   to recover the solution in the original (unscaled) variable space.

   :param model: Required Gurobi model to scale.
   :param method: Scaling method to use. One of:

                  ``'equilibration'``: iteratively scales rows and columns by their mean
                  absolute value. Supports LP, QP, and QCP models.

                  ``'geometric_mean'``: scales rows and columns by the geometric mean
                  of their coefficient ranges. Supports LP and QCP models.

                  ``'arithmetic_mean'``: scales rows and columns by the arithmetic
                  mean of their absolute values. Supports LP and QCP models.

   :param scale_passes: Maximum number of scaling iterations. Default: 5.
   :param scale_rel_tol: Relative convergence tolerance. Scaling stops early
                         when the improvement between iterations falls below
                         this threshold. Default: 1e-4.
   :param scaling_lb: Lower bound for scaling factors. Prevents extreme
                      downscaling. Default: 1e-8.
   :param scaling_ub: Upper bound for scaling factors. Prevents extreme
                      upscaling. Default: 1e8.
   :param value_threshold: Coefficients with absolute value below this
                           threshold are treated as zero. Default: 1e-13.
   :param scaling_time_limit: Time limit in seconds for the scaling
                              iterations. If reached, the best scaling found
                              so far is used. Default: no limit.
   :param scaling_log: Optional path to a log file. If provided, scaling
                       progress is written to this file. Default: no file.
   :param scaling_log_to_console: Set to 1 (default) to print scaling
                                  progress to the console, 0 to suppress.
   :param init_scaling: Controls use of user-provided initial scaling factors
                        set via the ``_init_scaling`` attribute on variables
                        and constraints:

                        ``0`` (default): ignore ``_init_scaling``; run the
                        iterative algorithm from the identity scaling.

                        ``1``: apply ``_init_scaling`` as the final scaling
                        and return immediately without running the iterative
                        algorithm.

                        ``2`` (warmstart): pre-apply ``_init_scaling``, then
                        run the iterative algorithm on top. The final factors
                        are the product of the user-provided values and the
                        algorithm's output.

   :param env: Optional Gurobi environment (``gurobipy.Env``) to use for
               the scaled model.
   :return: A :ref:`ScaledModel <APIScaledModelLabel>` object containing the
            scaled model with scaling information attached.


.. _APIScaledModelLabel:

ScaledModel
***********

``ScaledModel`` is a subclass of ``gurobipy.Model`` returned by
:py:func:`scale_model`. It adds methods and properties for recovering
unscaled solutions and computing violations in the original variable space.

.. py:method:: ScaledModel.getVarsUnscaled()

   Return a list of :ref:`ScaledVar <APIScaledVarLabel>` objects, one per
   variable in the model. Each object exposes both the scaled solution value
   (``X``) and the unscaled value (``Xunsc``). Must be called after
   optimization.

   :return: List of :class:`ScaledVar` objects.

.. py:method:: ScaledModel.getConstrsUnscaled()

   Return a list of :ref:`ScaledConstr <APIScaledConstrLabel>` objects, one
   per linear constraint. After calling :py:meth:`ComputeUnscVio`, each
   object exposes the unscaled constraint violation via ``UnscViolation``.

   :return: List of :class:`ScaledConstr` objects.

.. py:method:: ScaledModel.getQConstrsUnscaled()

   Return a list of :ref:`ScaledQConstr <APIScaledConstrLabel>` objects, one
   per quadratic constraint. After calling :py:meth:`ComputeUnscVio`, each
   object exposes the unscaled constraint violation via ``UnscViolation``.

   :return: List of :class:`ScaledQConstr` objects.

.. py:method:: ScaledModel.ComputeUnscVio(original_model)

   Compute constraint and bound violations in the original (unscaled) variable
   space. Must be called after optimization. Populates ``UnscViolation`` on
   all constraint wrappers and ``UnscBoundViolation`` on all variable wrappers,
   and sets the ``MaxUnscVio``, ``MaxUnscConstrVio``, and ``MaxUnscBoundVio``
   properties.

   :param original_model: The original (unscaled) Gurobi model.

.. py:method:: ScaledModel.ComputeUnscObj(original_model)

   Compute the objective value in the original (unscaled) variable space using
   the unscaled solution values from :py:meth:`getVarsUnscaled`. Must be
   called after optimization.

   :param original_model: The original (unscaled) Gurobi model.
   :return: The unscaled objective value as a float.

.. py:attribute:: ScaledModel.MaxUnscVio

   Maximum unscaled violation across all constraints and variable bounds.
   Available after calling :py:meth:`ComputeUnscVio`.

.. py:attribute:: ScaledModel.MaxUnscConstrVio

   Maximum unscaled violation across all linear and quadratic constraints.
   Available after calling :py:meth:`ComputeUnscVio`.

.. py:attribute:: ScaledModel.MaxUnscBoundVio

   Maximum unscaled variable bound violation.
   Available after calling :py:meth:`ComputeUnscVio`.

.. py:attribute:: ScaledModel.ScalingTime

   Wall-clock time in seconds taken by the scaling procedure.

.. py:attribute:: ScaledModel.ColScaling

   Diagonal column scaling matrix as a ``scipy.sparse`` matrix. Entry
   :math:`i` contains the scaling factor applied to variable :math:`i`.

.. py:attribute:: ScaledModel.RowScaling

   Diagonal row scaling matrix as a ``scipy.sparse`` matrix. Entry
   :math:`i` contains the scaling factor applied to constraint :math:`i`.


.. _APIScaledVarLabel:

ScaledVar
*********

Wrapper around a ``gurobipy.Var`` object returned by
:py:meth:`ScaledModel.getVarsUnscaled`. All standard Gurobi variable
attributes (e.g. ``VarName``, ``LB``, ``UB``) are forwarded to the
underlying variable.

.. py:attribute:: ScaledVar.X

   Solution value in the scaled model space.

.. py:attribute:: ScaledVar.Xunsc

   Solution value recovered in the original (unscaled) space:
   :math:`x_i = s_i \cdot y_i`, where :math:`s_i` is the column scaling
   factor and :math:`y_i` is the scaled solution value.

.. py:attribute:: ScaledVar.UnscBoundViolation

   Unscaled bound violation for this variable. Available after calling
   :py:meth:`ScaledModel.ComputeUnscVio`.


.. _APIScaledConstrLabel:

ScaledConstr / ScaledQConstr
****************************

Wrappers around ``gurobipy.Constr`` and ``gurobipy.QConstr`` objects returned
by :py:meth:`ScaledModel.getConstrsUnscaled` and
:py:meth:`ScaledModel.getQConstrsUnscaled` respectively. All standard
Gurobi constraint attributes are forwarded to the underlying object.

.. py:attribute:: ScaledConstr.UnscViolation
                  ScaledQConstr.UnscViolation

   Unscaled constraint violation. Available after calling
   :py:meth:`ScaledModel.ComputeUnscVio`.
