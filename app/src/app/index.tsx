import * as ImagePicker from 'expo-image-picker';
import { Link, router, Stack, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { checkHealth, predict } from '../lib/api';
import { prepareForUpload } from '../lib/image';
import { colors } from '../lib/theme';

type ServerState = { status: 'checking' } | { status: 'ok'; model: string } | { status: 'down' };

const PICKER_OPTIONS: ImagePicker.ImagePickerOptions = {
  mediaTypes: ['images'],
  allowsEditing: true, // iOS shows a square crop: the user centers the lesion, as the model expects
  quality: 1,
};

export default function Home() {
  const [server, setServer] = useState<ServerState>({ status: 'checking' });
  const [busy, setBusy] = useState(false);

  const refreshServer = useCallback(() => {
    setServer({ status: 'checking' });
    checkHealth()
      .then((h) => setServer({ status: 'ok', model: h.model }))
      .catch(() => setServer({ status: 'down' }));
  }, []);

  useFocusEffect(refreshServer);

  async function analyze(source: 'camera' | 'library') {
    const permission =
      source === 'camera'
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permiso necesario', 'Actívalo en Ajustes del iPhone para continuar.');
      return;
    }

    const result =
      source === 'camera'
        ? await ImagePicker.launchCameraAsync(PICKER_OPTIONS)
        : await ImagePicker.launchImageLibraryAsync(PICKER_OPTIONS);
    if (result.canceled) return;

    const asset = result.assets[0];
    setBusy(true);
    try {
      const uri = await prepareForUpload(asset.uri, asset.width, asset.height);
      const prediction = await predict(uri);
      router.push({ pathname: '/result', params: { imageUri: uri, data: JSON.stringify(prediction) } });
    } catch (e: any) {
      Alert.alert('No se pudo analizar', e?.message ?? String(e));
      refreshServer();
    } finally {
      setBusy(false);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Stack.Screen
        options={{
          headerRight: () => (
            <Link href="/settings" style={styles.headerLink}>
              Ajustes
            </Link>
          ),
        }}
      />

      <Pressable onPress={refreshServer} style={styles.status}>
        <View
          style={[
            styles.dot,
            {
              backgroundColor:
                server.status === 'ok' ? colors.ok : server.status === 'down' ? colors.danger : colors.muted,
            },
          ]}
        />
        <Text style={styles.statusText}>
          {server.status === 'checking' && 'Conectando con el servidor…'}
          {server.status === 'ok' && 'Servidor conectado'}
          {server.status === 'down' && 'Sin conexión con el servidor (toca para reintentar)'}
        </Text>
      </Pressable>

      <Text style={styles.title}>Analiza una lesión de la piel</Text>
      <Text style={styles.subtitle}>
        Haz una foto o elige una de tu galería. Después recórtala dejando la lesión en el centro.
      </Text>

      {busy ? (
        <View style={styles.busy}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.muted}>Analizando…</Text>
        </View>
      ) : (
        <View style={styles.actions}>
          <Pressable style={[styles.button, styles.primary]} onPress={() => analyze('camera')}>
            <Text style={styles.primaryText}>Hacer foto</Text>
          </Pressable>
          <Pressable style={[styles.button, styles.secondary]} onPress={() => analyze('library')}>
            <Text style={styles.secondaryText}>Elegir de la galería</Text>
          </Pressable>
        </View>
      )}

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Para una buena foto</Text>
        <Text style={styles.tip}>• Luz natural o buena iluminación, sin flash.</Text>
        <Text style={styles.tip}>• A unos 10 cm, con la lesión nítida y enfocada.</Text>
        <Text style={styles.tip}>• La lesión centrada y ocupando buena parte de la imagen.</Text>
      </View>

      <View style={[styles.card, styles.warning]}>
        <Text style={styles.cardTitle}>Prototipo de investigación</Text>
        <Text style={styles.tip}>
          No es un producto sanitario ni da diagnósticos. El modelo se ha entrenado con imágenes de
          dermatoscopio, así que con fotos de móvil sus resultados son poco fiables. Ante cualquier lesión
          que te preocupe o cambie, consulta a un dermatólogo.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, gap: 16 },
  headerLink: { color: colors.primary, fontSize: 16 },
  status: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  statusText: { color: colors.muted, fontSize: 14, flexShrink: 1 },
  title: { fontSize: 26, fontWeight: '700', color: colors.text, marginTop: 8 },
  subtitle: { fontSize: 16, color: colors.muted, lineHeight: 22 },
  actions: { gap: 12, marginVertical: 8 },
  button: { paddingVertical: 16, borderRadius: 12, alignItems: 'center' },
  primary: { backgroundColor: colors.primary },
  primaryText: { color: '#fff', fontSize: 17, fontWeight: '600' },
  secondary: { backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border },
  secondaryText: { color: colors.primary, fontSize: 17, fontWeight: '600' },
  busy: { alignItems: 'center', gap: 12, paddingVertical: 32 },
  muted: { color: colors.muted },
  card: { backgroundColor: colors.card, borderRadius: 12, padding: 16, gap: 6, borderWidth: 1, borderColor: colors.border },
  warning: { backgroundColor: '#FFF8E6', borderColor: '#F1D8A0' },
  cardTitle: { fontSize: 16, fontWeight: '600', color: colors.text, marginBottom: 4 },
  tip: { fontSize: 15, color: colors.text, lineHeight: 21 },
});
