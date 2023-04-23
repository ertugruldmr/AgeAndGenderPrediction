import gradio as gr
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np

# declerations
model_path = 'GenAgeModel' 
u_gif = "usage.gif"
gender_dict = {0:'Male', 1:'Female'}
examples_path = "samples"
usage_text = """- Focus the face.
              - upper side of image : forehead
              - lower side of image : chin
              - lef and right side border: ears"""


# laoding the model
model = tf.keras.models.load_model(model_path)


# Util Functions
def transmit(image):return image
def usage_gif():return u_gif

def pre_process(image):
  
  # Grayscaling
  image = ImageOps.grayscale(image)
  
  # resizing for the model
  image = image.resize((128, 128), Image.ANTIALIAS)
  
  # type casting to further processes
  image = np.array(image)

  # value scaling into 0-1 value range
  image = image/255.0

  return image

def predict(image):

  image = pre_process(image)

  # prediction
  pred = model.predict(image.reshape(1, 128, 128, 1))
  
  # extracting the results
  pred_gender = gender_dict[round(pred[0][0][0])]
  pred_age = round(pred[1][0][0])
  
  return pred_gender, pred_age

with gr.Blocks() as demo:
    # defining the components
    gr.Markdown("Use the __capture icon__ where on the __bottom center of the camera window__ for __taking photo__")
    gr.Markdown("For better prediction, pose like examples. Get close the camera, open the lights etc...")
    gr.Markdown(usage_text)
    with gr.Row():
      image = gr.Image(value=u_gif, shape=(224, 224), type="pil",abel="upload images")
      cam_image = gr.Image(shape=(224, 224), type="pil", source="webcam", label="Take Photo Via ICON")
    with gr.Row():
      predict_btn = gr.Button("Predict Gender and Age")
      capture_btn = gr.Button("Set the Captured Image")
    usage_btn = gr.Button("Show Usage Gif for meaningful predictions")
    gender = gr.Textbox(label="Predicted Gender")
    age = gr.Textbox(label="Predicted Age")
    
    # setting the functions
    
    predict_btn.click(fn=predict, inputs=image, outputs=[gender, age])
    capture_btn.click(fn=transmit, inputs=cam_image, outputs=image)
    usage_btn.click(fn=usage_gif, inputs=None, outputs=image)
      
    # adding the examples
    gr.Examples(examples_path, inputs=[image])

# Launching the demo
if __name__ == "__main__":
    demo.launch()
