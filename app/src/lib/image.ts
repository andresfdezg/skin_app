import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';

const MAX_SIDE = 1024;

/** Downscale large photos and re-encode as JPEG so uploads are fast and always decodable by the API. */
export async function prepareForUpload(uri: string, width: number, height: number): Promise<string> {
  const context = ImageManipulator.manipulate(uri);
  if (Math.max(width, height) > MAX_SIDE) {
    context.resize(width >= height ? { width: MAX_SIDE } : { height: MAX_SIDE });
  }
  const image = await context.renderAsync();
  const saved = await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.9 });
  return saved.uri;
}
