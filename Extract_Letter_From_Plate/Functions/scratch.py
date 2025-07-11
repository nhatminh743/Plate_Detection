import keras

model_dir = r'/home/minhpn/Desktop/Green_Parking/CNN_Recognize/CNN_Model.keras'

model = keras.model.load_model(model_dir)

prediction = model.predict('/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Extracted_Letter/0229_05817_b_plate/ROI_0.jpg')