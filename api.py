from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from skimage import io
import torch
from PIL import Image
from io import BytesIO
import base64
from npuengine import EngineOV
import numpy as np
from utilities import preprocess_image, postprocess_image

app = FastAPI()

# Initialize model
net = EngineOV("./models/rmbg.bmodel", device_id=0)

class ImageData(BaseModel):
    image_base64: str

def remove_background(image_data):
    model_input_size = [1024, 1024]
    orig_im = io.imread(image_data)
    orig_im_size = orig_im.shape[:2]
    image = preprocess_image(orig_im, model_input_size)
    
    # Inference
    result = net([image.numpy()])[0]
    result = torch.from_numpy(result).float()
    
    # Post-process
    result_image = postprocess_image(result, orig_im_size)
    
    # Creating no background image
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(image_data)
    no_bg_image.paste(orig_image, mask=pil_im)
    
    # Convert to bytes for output
    img_byte_arr = BytesIO()
    no_bg_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

@app.post("/remove-bg/")
async def remove_bg(data: ImageData):
    try:
        # Decode the base64 string to bytes
        image_data = base64.b64decode(data.image_base64)
        processed_image = remove_background(BytesIO(image_data))
        return StreamingResponse(BytesIO(processed_image), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

