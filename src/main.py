import io
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import StreamingResponse

import matplotlib.pyplot as plt
import uvicorn

# Загружаем модель машинного обучения
model = joblib.load("/home/bmf/Desktop/freelance/FastApi.ml.antifraud/models/LGBM.pickle")

app = FastAPI()

@app.post("/predict/")
async def upload_file(file: UploadFile = File(...)):
    # Читаем данные из загруженного файла
    contents = await file.read()
    # Создаем DataFrame из данных в файле
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    df = df.drop(['Class'], axis=1)
    # Используем модель для предсказания
    prediction = model.predict(df[:10])
    prediction = np.round(prediction)
    # Возвращаем результаты предсказания
    return {"prediction": prediction.tolist()}


@app.post("/plot/")
async def plot_graph(df_file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await df_file.read()))
    df = df.iloc[:, :-1]  # удаляем последний столбец
    df = df[:100]
    fig, axs = plt.subplots(10, 3, figsize=(15, 40))
    for i, ax in enumerate(axs.flat):
        ax.hist(df.iloc[:, i])
        ax.set_title(df.columns[i], fontsize=18, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('Значения', fontsize=15)
        ax.set_ylabel('Частота', fontsize=15)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)