import AsyncStorage from '@react-native-async-storage/async-storage';
import { File } from 'expo-file-system';
import { fetch } from 'expo/fetch';

const STORAGE_KEY = 'apiUrl';
const DEFAULT_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://192.168.1.131:8000';

export type Prediction = {
  prediction: { class: string; name: string; probability: number };
  probabilities: Record<string, number>;
  malignant_probability: number;
  model: string;
  inference_ms: number;
  disclaimer: string;
};

export type Health = { status: string; model: string; device: string };

export async function getApiUrl(): Promise<string> {
  return (await AsyncStorage.getItem(STORAGE_KEY)) ?? DEFAULT_URL;
}

export async function setApiUrl(url: string | null): Promise<void> {
  if (url) await AsyncStorage.setItem(STORAGE_KEY, url.trim().replace(/\/+$/, ''));
  else await AsyncStorage.removeItem(STORAGE_KEY);
}

export function defaultApiUrl(): string {
  return DEFAULT_URL;
}

async function request(path: string, init: RequestInit = {}, timeoutMs = 20000, baseUrl?: string) {
  const url = `${baseUrl ?? (await getApiUrl())}${path}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal } as any);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail ?? `El servidor respondió ${res.status}`);
    }
    return res.json();
  } catch (e: any) {
    if (e?.name === 'AbortError') throw new Error('El servidor no responde. ¿Está encendido y en la misma WiFi?');
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export function checkHealth(baseUrl?: string): Promise<Health> {
  return request('/health', {}, 4000, baseUrl);
}

export function predict(imageUri: string): Promise<Prediction> {
  const form = new FormData();
  form.append('file', new File(imageUri) as unknown as Blob);
  return request('/predict', { method: 'POST', body: form });
}
