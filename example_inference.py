from skimage import io
import torch, os
from PIL import Image
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from huggingface_hub import hf_hub_download
from npuengine import EngineOV
import numpy as np
import pdb
def example_inference():

    im_path = f"{os.path.dirname(os.path.abspath(__file__))}/example_input.jpg"

    #net = BriaRMBG()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
   # net.to(device)
   # net.eval()    
    
    net = EngineOV("./models/rmbg.bmodel",device_id=0)
    
    # prepare input
    model_input_size = [1024,1024]
    orig_im = io.imread(im_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size)

    # inference 
    result=net([image.numpy()])[0]
    result = torch.from_numpy(result).float() 
    # post process
    #pdb.set_trace()    
    result_image = postprocess_image(result, orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
    orig_image = Image.open(im_path)
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image.save("example_image_no_bg1.png")


if __name__ == "__main__":
    example_inference()
