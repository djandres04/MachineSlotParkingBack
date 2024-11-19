from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware

# Inicializar la app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (puedes restringir esto más adelante)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Cargar el modelo entrenado
model = YOLO("best.pt")

# Modelo para configuraciones
class DetectionSettings(BaseModel):
    conf_thres: float = 0.5  # Umbral de confianza
    iou_thres: float = 0.5   # Umbral de solapamiento

# Ruta para procesar imágenes
@app.post("/detect/")
async def detect_objects(
    file: UploadFile = File(...), settings: DetectionSettings = DetectionSettings()
):
    try:
        # Leer la imagen subida
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Realizar predicción con el modelo cargado
        results = model.predict(
            source=image,
            conf=settings.conf_thres,
            iou=settings.iou_thres,
            #device=0  # Usa la GPU si está disponible
        )

        # Convertir los resultados en un formato JSON legible
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "box": [float(x) for x in box.xyxy[0].tolist()]
                })

        return {"detections": detections}


    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))