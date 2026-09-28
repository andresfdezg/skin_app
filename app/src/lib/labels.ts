// Class codes returned by the API (HAM10000) and how we show them to the user.
export const CLASS_LABELS: Record<string, string> = {
  akiec: 'Queratosis actínica / carcinoma intraepitelial',
  bcc: 'Carcinoma basocelular',
  bkl: 'Queratosis benigna',
  df: 'Dermatofibroma',
  mel: 'Melanoma',
  nv: 'Nevus (lunar)',
  vasc: 'Lesión vascular',
};

export const MALIGNANT = new Set(['akiec', 'bcc', 'mel']);

export function labelFor(code: string): string {
  return CLASS_LABELS[code] ?? code;
}
