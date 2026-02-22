import type { ProjectionMethod, ProjectorParams } from "../engine/projection";
import {
  ACTIVE_SET_EPS,
  ALM_DUAL_RELAXATION,
  ALM_LAMBDA_CLIP,
  ALM_MAX_BACKTRACKS,
  ALM_MAX_STEP_NORM,
  ALM_MIN_STEP_SCALE,
  CONVEX_HALFSPACE_EPS,
  DEFAULT_DAMPING,
  LEGACY_STEP_CAP_RATIO,
  PROXIMAL_WEIGHT_DEFAULT,
} from "../engine/math/constants";

type HardProjectMode = "iters" | "tol";

export type ProjectionSettings = {
  method: ProjectionMethod;
  rho: number;
  wFree: number;
  wHandle: number;
  useVolumeConstraint: boolean;
  goalVolume: number;
  itersPerFrame: number;
  itersOnRelease: number;
  hardProjectMode: HardProjectMode;
  hardProjectMaxIters: number;
  hardProjectTolPlanar: number;

  qcqpDamping: number;

  almProximalWeight: number;
  almActiveSetEps: number;
  almMaxStepNorm: number;
  almMinStepScale: number;
  almMaxBacktracks: number;
  almDualRelaxation: number;
  almLambdaClip: number;

  convexHalfspaceEps: number;
  legacyStepCapRatio: number;
};

type ProjectionSettingId = Exclude<keyof ProjectionSettings, "method">;

type SettingOption<TValue extends string> = {
  value: TValue;
  label: string;
};

export type ProjectionSettingField = {
  id: ProjectionSettingId;
  label: string;
  input: "number" | "integer" | "select" | "boolean";
  min?: number;
  step?: number | "any";
  advanced?: boolean;
  methods?: ProjectionMethod[];
  options?: Array<SettingOption<HardProjectMode>>;
};

const guidedAlmMethods: ProjectionMethod[] = ["guided_alm_planar", "guided_alm_convex"];
const guidedAlmLegacyMethods: ProjectionMethod[] = ["guided_alm_legacy_planar", "guided_alm_legacy_convex"];
const consensusQcqpMethods: ProjectionMethod[] = ["consensus_qcqp_planar", "consensus_qcqp_convex_direct"];
const volumeAwareMethods: ProjectionMethod[] = [...guidedAlmMethods, ...consensusQcqpMethods];

export const projectionSettingFields: ProjectionSettingField[] = [
  { id: "rho", label: "rho", input: "number", min: 1e-12, step: "any" },
  { id: "wFree", label: "wFree", input: "number", min: 0, step: "any" },
  { id: "wHandle", label: "wHandle", input: "number", min: 0, step: "any" },
  { id: "useVolumeConstraint", label: "use volume constraint", input: "boolean", methods: volumeAwareMethods },
  { id: "goalVolume", label: "goal volume", input: "number", step: "any", methods: volumeAwareMethods },
  { id: "itersPerFrame", label: "iters/frame", input: "integer", min: 1, step: 1 },
  { id: "itersOnRelease", label: "iters/release", input: "integer", min: 1, step: 1 },
  {
    id: "hardProjectMode",
    label: "hard project mode",
    input: "select",
    options: [
      { value: "iters", label: "fixed iterations" },
      { value: "tol", label: "until tolerance" },
    ],
  },
  { id: "hardProjectMaxIters", label: "hard project max iters", input: "integer", min: 1, step: 1 },
  { id: "hardProjectTolPlanar", label: "hard project tolerance", input: "number", min: 0, step: "any" },

  {
    id: "qcqpDamping",
    label: "QCQP damping",
    input: "number",
    min: 0,
    step: "any",
    advanced: true,
    methods: consensusQcqpMethods,
  },

  {
    id: "almProximalWeight",
    label: "ALM proximal weight",
    input: "number",
    min: 0,
    step: "any",
    advanced: true,
    methods: guidedAlmMethods,
  },
  {
    id: "almActiveSetEps",
    label: "ALM active-set eps",
    input: "number",
    min: 0,
    step: "any",
    advanced: true,
    methods: guidedAlmMethods,
  },
  {
    id: "almMaxStepNorm",
    label: "ALM max step norm",
    input: "number",
    min: 1e-8,
    step: "any",
    advanced: true,
    methods: guidedAlmMethods,
  },
  {
    id: "almMinStepScale",
    label: "ALM min step scale",
    input: "number",
    min: 1e-8,
    step: "any",
    advanced: true,
    methods: guidedAlmMethods,
  },
  {
    id: "almMaxBacktracks",
    label: "ALM max backtracks",
    input: "integer",
    min: 0,
    step: 1,
    advanced: true,
    methods: guidedAlmMethods,
  },
  {
    id: "almDualRelaxation",
    label: "ALM dual relaxation",
    input: "number",
    min: 0,
    step: "any",
    advanced: true,
    methods: guidedAlmMethods,
  },
  {
    id: "almLambdaClip",
    label: "ALM lambda clip",
    input: "number",
    min: 1,
    step: "any",
    advanced: true,
    methods: guidedAlmMethods,
  },

  {
    id: "convexHalfspaceEps",
    label: "convex halfspace eps",
    input: "number",
    min: 0,
    step: "any",
    advanced: true,
    methods: ["convex", "guided_alm_legacy_convex"],
  },
  {
    id: "legacyStepCapRatio",
    label: "legacy step cap ratio",
    input: "number",
    min: 1e-8,
    step: "any",
    advanced: true,
    methods: guidedAlmLegacyMethods,
  },
];

export function createDefaultProjectionSettings(goalVolume = 1): ProjectionSettings {
  return {
    method: "guided_alm_convex",
    rho: 10,
    wFree: 1,
    wHandle: 1e5,
    useVolumeConstraint: true,
    goalVolume,
    itersPerFrame: 10,
    itersOnRelease: 120,
    hardProjectMode: "iters",
    hardProjectMaxIters: 400,
    hardProjectTolPlanar: 1e-6,

    qcqpDamping: DEFAULT_DAMPING,

    almProximalWeight: PROXIMAL_WEIGHT_DEFAULT,
    almActiveSetEps: ACTIVE_SET_EPS,
    almMaxStepNorm: ALM_MAX_STEP_NORM,
    almMinStepScale: ALM_MIN_STEP_SCALE,
    almMaxBacktracks: ALM_MAX_BACKTRACKS,
    almDualRelaxation: ALM_DUAL_RELAXATION,
    almLambdaClip: ALM_LAMBDA_CLIP,

    convexHalfspaceEps: CONVEX_HALFSPACE_EPS,
    legacyStepCapRatio: LEGACY_STEP_CAP_RATIO,
  };
}

export function isProjectionFieldVisible(field: ProjectionSettingField, method: ProjectionMethod, showAdvanced: boolean): boolean {
  if (!showAdvanced && field.advanced) return false;
  if (!field.methods || field.methods.length === 0) return true;
  return field.methods.includes(method);
}

export function toProjectorParams(settings: ProjectionSettings): ProjectorParams {
  return {
    rho: settings.rho,
    wFree: settings.wFree,
    wHandle: settings.wHandle,
    useVolumeConstraint: settings.useVolumeConstraint,
    goalVolume: settings.goalVolume,
    itersPerFrame: settings.itersPerFrame,
    itersOnRelease: settings.itersOnRelease,

    qcqpDamping: settings.qcqpDamping,

    almProximalWeight: settings.almProximalWeight,
    almActiveSetEps: settings.almActiveSetEps,
    almMaxStepNorm: settings.almMaxStepNorm,
    almMinStepScale: settings.almMinStepScale,
    almMaxBacktracks: settings.almMaxBacktracks,
    almDualRelaxation: settings.almDualRelaxation,
    almLambdaClip: settings.almLambdaClip,

    convexHalfspaceEps: settings.convexHalfspaceEps,
    legacyStepCapRatio: settings.legacyStepCapRatio,
  };
}
