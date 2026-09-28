import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

import { colors } from '../lib/theme';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerTintColor: colors.primary,
          headerTitleStyle: { color: colors.text },
          contentStyle: { backgroundColor: colors.background },
        }}
      >
        <Stack.Screen name="index" options={{ title: 'vibe_skin' }} />
        <Stack.Screen name="result" options={{ title: 'Resultado' }} />
        <Stack.Screen name="settings" options={{ title: 'Ajustes', presentation: 'modal' }} />
      </Stack>
    </>
  );
}
