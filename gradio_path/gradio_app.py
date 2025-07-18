
#-------------------------------------------- IMPORT LIBRARIES ---------------------------------------------------------
import gradio as gr
import requests


#-------------------------------------------------  EXTERNAL LINKS  ----------------------------------------------------
css_fileLink = r'/home/minhpn/Desktop/Green_Parking/gradio_path/styles.css'
#--------------------------------------------------- FUNCTIONS ---------------------------------------------------------
def upload_and_return_prediction(file_paths):
    url = "http://127.0.0.1:8000/upload-files"

    files = [("files", (file_path.split("/")[-1], open(file_path, "rb"), "image/jpeg")) for file_path in file_paths]

    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.json()

        if "results" in data:
            result_lines = []
            for filename, info in data["results"].items():
                result_lines.append(f"{filename}: {info['text']}")
            return "\n".join(result_lines)
        else:
            return "❌ No results found."

    except Exception as e:
        return f"❌ Error: {str(e)}"

#---------------------------------------------------- GUI DESIGN -------------------------------------------------------

demo = gr.Interface(
    fn=upload_and_return_prediction,
    inputs=gr.File(
        label="Upload license plate image(s)",
        type="filepath",
        file_count='multiple',
        height=500,
    ),
    outputs=gr.Textbox(label="Detected Text"),
    title="License Plate Recognition"
)

#---------------------------------------------------- DEPLOYMENT -------------------------------------------------------
if __name__ == "__main__":
    demo.launch()

