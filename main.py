from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import io
import cv2
import supervision as sv
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

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
model = YOLO("weight/v1440-100/best.pt")

# Modelo para configuraciones
class DetectionSettings(BaseModel):
    conf_thres: float = 0.5  # Umbral de confianza
    iou_thres: float = 0.8   # Umbral de solapamiento

# Ruta para procesar imágenes
@app.post("/detect/")
async def detect_objects(
    file: UploadFile = File(...), settings: DetectionSettings = DetectionSettings()
):
    # Validar que el archivo es de tipo imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe ser una imagen válida (ejemplo: image/jpeg, image/png)."
        )
    try:
        # Leer la imagen subida
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo abrir el archivo como una imagen válida."
            )

        # Convertir la imagen a formato OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Realizar predicción con el modelo cargado
        results = model(image_cv, conf=settings.conf_thres, iou=settings.iou_thres)[0]  
        detections = sv.Detections.from_ultralytics(results)

        # Anotar la imagen
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]

        annotated_frame = image_cv.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        # Convertir la imagen anotada a un objeto binario
        img_byte_array = io.BytesIO()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
        Image.fromarray(annotated_frame).save(img_byte_array, format="JPEG")
        img_byte_array.seek(0)

        # Retornar la imagen procesada
        return StreamingResponse(img_byte_array, media_type="image/jpeg")

    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))