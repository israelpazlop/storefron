from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from io import BytesIO
import base64
import mimetypes

app = FastAPI()

# 🔑 Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción cambiar por tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key="AIzaSyDNZuwyfS0lVv70UwsvM5zdk0BodEu8kwQ")


# 🔹 Codificar la imagen subida en base64
async def encode_upload(file: UploadFile):
    mime_type, _ = mimetypes.guess_type(file.filename)
    data_bytes = await file.read()
    data = base64.b64encode(data_bytes).decode("utf-8")
    return {"inline_data": {"mime_type": mime_type or "image/png", "data": data}}


@app.post("/combine")
async def combine_images(
    dress: UploadFile = File(...),
    model: UploadFile = File(...),
):
    try:
        # ⚡ Codificar imágenes de forma async
        model_part = await encode_upload(model)  # mujer primero
        dress_part = await encode_upload(dress)  # vestido segundo

        prompt = {
            "text": """Create a professional e-commerce fashion photo.
The woman from the first image must wear the dress from the second image.
Generate a realistic full-body fashion photo with natural shadows and lighting.
Preserve the exact face, identity, and hairstyle of the woman.
Do not redraw, replace, or alter the face in any way.
Only change the clothing and adjust body proportions to fit the dress."""
        }

        # Llamada al modelo
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[model_part, dress_part, prompt],
        )

        # 🔹 Extraer imagen de forma segura
        first_candidate = response.candidates[0]
        raw_bytes = None
        mime_type = "image/png"

        for part in first_candidate.content.parts:
            if getattr(part, "inline_data", None):
                data = part.inline_data.data
                # Si viene con prefijo "data:image/png;base64,..."
                if isinstance(data, str):
                    if "," in data:
                        mime_type = data.split(";")[0].replace("data:", "")
                        data = data.split(",")[1]
                    raw_bytes = base64.b64decode(data)
                else:  # si ya es bytes
                    raw_bytes = data
                break

        if not raw_bytes:
            return JSONResponse(
                {"error": "No se generó ninguna imagen"}, status_code=500
            )

        return StreamingResponse(BytesIO(raw_bytes), media_type=mime_type)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
