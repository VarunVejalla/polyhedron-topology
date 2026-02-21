import { projectionMethods, type ProjectionMethod } from "../engine/projection";
import {
  isProjectionFieldVisible,
  projectionSettingFields,
  type ProjectionSettingField,
  type ProjectionSettings,
} from "../state/projectionSettings";

type Props = {
  value: ProjectionSettings;
  showAdvanced: boolean;
  onPatch: (patch: Partial<ProjectionSettings>) => void;
  onShowAdvancedChange: (next: boolean) => void;
};

function clampNumeric(value: number, field: ProjectionSettingField): number {
  if (!Number.isFinite(value)) return NaN;
  if (field.input === "integer") value = Math.round(value);
  if (field.min !== undefined) value = Math.max(field.min, value);
  return value;
}

export function ProjectionSettingsPanel({ value, showAdvanced, onPatch, onShowAdvancedChange }: Props) {
  const visibleFields = projectionSettingFields.filter((field) => isProjectionFieldVisible(field, value.method, showAdvanced));

  return (
    <div className="settingsPanel">
      <div className="settingsHeader">
        <label className="toolbarField">
          Method
          <select value={value.method} onChange={(e) => onPatch({ method: e.target.value as ProjectionMethod })}>
            {projectionMethods.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <label className="settingsToggle">
          <input type="checkbox" checked={showAdvanced} onChange={(e) => onShowAdvancedChange(e.target.checked)} />
          Show advanced
        </label>
      </div>

      <details className="settingsDropdown">
        <summary>Editable parameters ({visibleFields.length})</summary>
        <div className="settingsGrid">
          {visibleFields.map((field) => {
            const key = field.id;
            const current = value[key];
            if (field.input === "select") {
              return (
                <label key={field.id} className="toolbarField">
                  {field.label}
                  <select
                    value={String(current)}
                    onChange={(e) => onPatch({ [key]: e.target.value } as Partial<ProjectionSettings>)}
                  >
                    {field.options?.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
              );
            }
            return (
              <label key={field.id} className="toolbarField">
                {field.label}
                <input
                  type="number"
                  min={field.min}
                  step={field.step ?? "any"}
                  value={typeof current === "number" ? current : 0}
                  onChange={(e) => {
                    const parsed = clampNumeric(Number(e.target.value), field);
                    if (!Number.isFinite(parsed)) return;
                    onPatch({ [key]: parsed } as Partial<ProjectionSettings>);
                  }}
                />
              </label>
            );
          })}
        </div>
      </details>
    </div>
  );
}
