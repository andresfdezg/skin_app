export const colors = {
  background: '#F6F7F9',
  card: '#FFFFFF',
  text: '#1B1F24',
  muted: '#5F6B7A',
  border: '#E1E5EA',
  primary: '#2F5DDB',
  ok: '#1F8A4C',
  warn: '#B7791F',
  danger: '#C53030',
};

/** Color for a malignant probability: green below 30%, amber up to 60%, red above. */
export function riskColor(p: number): string {
  if (p < 0.3) return colors.ok;
  if (p < 0.6) return colors.warn;
  return colors.danger;
}
