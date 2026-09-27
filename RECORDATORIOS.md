# Recordatorios y tareas manuales

Tareas que tengo que hacer yo por mi cuenta durante el proyecto, más ideas aparcadas para después.
Marca con `[x]` lo que esté hecho.

## Pendiente ahora

- [X] **Datasets fuera de OneDrive.** Crear `C:\datasets\` y guardar ahí todas las imágenes. Nunca dentro de la carpeta del proyecto.
- [X] **Decidir si muevo el proyecto fuera de OneDrive** (p. ej. a `C:\Proyectos\vibe_skin`), con GitHub como copia de seguridad. Motivo: `.venv`, checkpoints y la carpeta `.git` generan miles de archivos que OneDrive sincroniza mal.
- [X] **Descargar HAM10000** desde Kaggle (hace falta cuenta) y descomprimirlo en `C:\datasets\ham10000\`. Pasarle a Claude la ruta.
- [X] Crear un repositorio en GitHub (privado) para el proyecto.
- [X] Copiar la cita exacta de Dataverse en `DATA_LICENSE.md`.
- [ ] (Opcional) Comprobar si en el zip del test de ISIC 2018 está `ISIC_0035068.jpg`. Falta en `datasets/ham10000/ISIC2018_Task3_Test_Images/`; de momento el pipeline la descarta (evaluamos con 1.511 de 1.512 imágenes).

> Nota: al final los datasets están en `C:\Proyectos\vibe_skin\datasets\` (dentro del proyecto pero fuera de OneDrive). La carpeta está en `.gitignore`, así que nunca se sube a GitHub.

## Más adelante (fase app)

- [ ] Instalar **Expo Go** en el iPhone.
- [ ] Permitir la conexión en el **firewall de Windows** para el puerto de la API la primera vez que se arranque.
- [ ] Aceptar el permiso de **"red local"** que pedirá iOS la primera vez.

## Datasets en la recámara

Por ahora solo usamos HAM10000. Los demás se incorporan en iteraciones posteriores:

| Dataset | Tipo de imagen | Rol en el proyecto | Cuándo |
|---|---|---|---|
| **HAM10000** | Dermatoscopia | Baseline del MVP; validar que el pipeline funciona de principio a fin | Ahora |
| **PAD-UFES-20** | Fotos de smartphone | Adaptar el modelo al dominio real de la app (fotos de móvil, no dermatoscopio) | Iteración 1 |
| **ISIC 2019/2020** | Dermatoscopia | Ampliar volumen de datos y clases; mejorar la robustez general | Iteración 2+ |
| **Fitzpatrick17k / DDI** | Clínica, varios fototipos | Evaluar sesgos: comprobar que el modelo funciona igual en pieles claras y oscuras (sobre todo para evaluación, no para entrenar) | Iteración 2+ |

## Ideas aparcadas

- Rechazo de fotos inválidas (borrosas, sin lesión, que no son piel).
- Calibración de probabilidades (temperature scaling).
- Grad-CAM para mostrar qué zona ha mirado el modelo.
- Inferencia en el propio iPhone (CoreML) en lugar de en el servidor.
- Acceso a la API fuera de casa (Tailscale) sin montar un servidor en la nube.
- Aviso legal en la app: herramienta de apoyo/triaje, **no diagnóstico**.
