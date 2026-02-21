// Centralized numeric tolerances for the engine (determinism + easy tuning).
export const EPS = 1e-12;
export const EPS_AREA = 1e-14; // degeneracy checks on areas / covariance magnitudes

// Generic solver conditioning constants.
export const MIN_RHO = 1e-8;
export const MIN_RHO_LEGACY = 1e-6;
export const SOLVER_PIVOT_EPS = 1e-14;
export const SOLVER_FALLBACK_DIAG = 1e-4;
export const DEFAULT_DAMPING = 1e-6;
export const ACTIVE_SET_EPS = 1e-10;
export const INVERSE_DENOM_EPS = 1e-12;
export const INVERSE_DENOM_EPS_LOOSE = 1e-10;

// ALM tuning.
export const ALM_MAX_STEP_NORM = 0.5;
export const ALM_MIN_STEP_SCALE = 1 / 64;
export const ALM_MAX_BACKTRACKS = 8;
export const ALM_DUAL_RELAXATION = 0.25;
export const ALM_LAMBDA_CLIP = 1e6;
export const BACKTRACK_SHRINK = 0.5;
export const PROXIMAL_WEIGHT_DEFAULT = 1e-6;

// QCQP-1 paper-style local solve tuning.
export const QCQP_EIG_TOL = 1e-12;
export const QCQP_DENOM_EPS = 1e-14;
export const QCQP_ROOT_EPS = 1e-10;
export const QCQP_ROOT_WIDTH_EPS = 1e-12;
export const QCQP_BRACKET_EXPANSIONS = 64;
export const QCQP_BISECTION_ITERS = 128;

// Projection / geometry constants.
export const CONVEX_HALFSPACE_EPS = 1e-6;
export const LEGACY_STEP_CAP_RATIO = 0.2;
export const BBOX_MIN_SCALE = 1e-6;
export const PLANE_VARIABLE_REGULARIZATION = 1e-3;
export const POLYGON_INSIDE_EPS = 1e-9;

// Canonical builder defaults.
export const CANONICAL_TOL_DEFAULT = 1e-7;
export const CANONICAL_2D_SCALE_EPS = 1e-9;
