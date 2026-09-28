import { router } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

import { checkHealth, defaultApiUrl, getApiUrl, setApiUrl } from '../lib/api';
import { colors } from '../lib/theme';

export default function Settings() {
  const [url, setUrl] = useState('');
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    getApiUrl().then(setUrl);
  }, []);

  async function test() {
    setTesting(true);
    setMessage(null);
    try {
      const h = await checkHealth(url.trim().replace(/\/+$/, ''));
      setMessage({ ok: true, text: `Conectado. Modelo ${h.model} (${h.device}).` });
    } catch (e: any) {
      setMessage({ ok: false, text: e?.message ?? 'No se pudo conectar.' });
    } finally {
      setTesting(false);
    }
  }

  async function save() {
    await setApiUrl(url);
    router.back();
  }

  async function reset() {
    await setApiUrl(null);
    setUrl(defaultApiUrl());
    setMessage(null);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Dirección del servidor</Text>
      <TextInput
        value={url}
        onChangeText={setUrl}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="http://192.168.1.131:8000"
        style={styles.input}
      />
      <Text style={styles.help}>
        El PC y el iPhone tienen que estar en la misma WiFi. En el PC: uv run uvicorn api.main:app --host
        0.0.0.0 --port 8000
      </Text>

      <View style={styles.actions}>
        <Pressable style={[styles.button, styles.secondary]} onPress={test} disabled={testing}>
          {testing ? <ActivityIndicator color={colors.primary} /> : <Text style={styles.secondaryText}>Probar conexión</Text>}
        </Pressable>
        <Pressable style={[styles.button, styles.primary]} onPress={save}>
          <Text style={styles.primaryText}>Guardar</Text>
        </Pressable>
      </View>

      {message && <Text style={{ color: message.ok ? colors.ok : colors.danger, fontSize: 15 }}>{message.text}</Text>}

      <Pressable onPress={reset}>
        <Text style={styles.reset}>Volver a la dirección por defecto</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, gap: 12 },
  label: { fontSize: 14, color: colors.muted },
  input: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 10,
    padding: 14,
    fontSize: 16,
    color: colors.text,
  },
  help: { fontSize: 13, color: colors.muted, lineHeight: 18 },
  actions: { flexDirection: 'row', gap: 12, marginTop: 8 },
  button: { flex: 1, paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  primary: { backgroundColor: colors.primary },
  primaryText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  secondary: { backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border },
  secondaryText: { color: colors.primary, fontSize: 16, fontWeight: '600' },
  reset: { color: colors.primary, fontSize: 15, marginTop: 8 },
});
