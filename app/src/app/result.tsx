import { router, useLocalSearchParams } from 'expo-router';
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import type { Prediction } from '../lib/api';
import { labelFor, MALIGNANT } from '../lib/labels';
import { colors, riskColor } from '../lib/theme';

const pct = (p: number) => `${Math.round(p * 100)}%` as const;

export default function Result() {
  const { imageUri, data } = useLocalSearchParams<{ imageUri: string; data: string }>();
  const result: Prediction = JSON.parse(data);
  const malignant = result.malignant_probability;
  const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Image source={{ uri: imageUri }} style={styles.image} />

      <View style={styles.card}>
        <Text style={styles.label}>Probabilidad de lesión maligna según el modelo</Text>
        <Text style={[styles.big, { color: riskColor(malignant) }]}>{pct(malignant)}</Text>
        <View style={styles.track}>
          <View style={[styles.fill, { width: pct(malignant), backgroundColor: riskColor(malignant) }]} />
        </View>
        <Text style={styles.small}>Suma de melanoma, carcinoma basocelular y queratosis actínica.</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Tipo de lesión más probable</Text>
        <Text style={styles.top}>{labelFor(result.prediction.class)}</Text>
        <Text style={styles.small}>Confianza del modelo: {pct(result.prediction.probability)}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Todas las clases</Text>
        {ranked.map(([code, p]) => (
          <View key={code} style={styles.row}>
            <View style={styles.rowHeader}>
              <Text style={styles.rowName}>
                {labelFor(code)}
                {MALIGNANT.has(code) ? '  ·  maligna' : ''}
              </Text>
              <Text style={styles.rowPct}>{pct(p)}</Text>
            </View>
            <View style={styles.trackThin}>
              <View
                style={[styles.fill, { width: pct(p), backgroundColor: MALIGNANT.has(code) ? colors.danger : colors.primary }]}
              />
            </View>
          </View>
        ))}
      </View>

      <View style={[styles.card, styles.warning]}>
        <Text style={styles.tip}>
          Esto no es un diagnóstico. Es un prototipo de investigación entrenado con imágenes de
          dermatoscopio. Consulta a un dermatólogo ante cualquier lesión que te preocupe o cambie.
        </Text>
      </View>

      <Text style={styles.meta}>
        Modelo {result.model} · {Math.round(result.inference_ms)} ms
      </Text>

      <Pressable style={styles.button} onPress={() => router.back()}>
        <Text style={styles.buttonText}>Analizar otra</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, gap: 16 },
  image: { width: '100%', aspectRatio: 1, borderRadius: 12, backgroundColor: colors.border },
  card: { backgroundColor: colors.card, borderRadius: 12, padding: 16, gap: 8, borderWidth: 1, borderColor: colors.border },
  warning: { backgroundColor: '#FFF8E6', borderColor: '#F1D8A0' },
  label: { fontSize: 14, color: colors.muted },
  big: { fontSize: 40, fontWeight: '700' },
  top: { fontSize: 22, fontWeight: '600', color: colors.text },
  small: { fontSize: 13, color: colors.muted },
  track: { height: 10, borderRadius: 5, backgroundColor: colors.border, overflow: 'hidden' },
  trackThin: { height: 6, borderRadius: 3, backgroundColor: colors.border, overflow: 'hidden' },
  fill: { height: '100%' },
  row: { gap: 4, marginTop: 4 },
  rowHeader: { flexDirection: 'row', justifyContent: 'space-between', gap: 8 },
  rowName: { fontSize: 15, color: colors.text, flexShrink: 1 },
  rowPct: { fontSize: 15, color: colors.text, fontVariant: ['tabular-nums'] },
  tip: { fontSize: 14, color: colors.text, lineHeight: 20 },
  meta: { fontSize: 12, color: colors.muted, textAlign: 'center' },
  button: { backgroundColor: colors.primary, paddingVertical: 16, borderRadius: 12, alignItems: 'center', marginBottom: 24 },
  buttonText: { color: '#fff', fontSize: 17, fontWeight: '600' },
});
