from ultralytics import YOLO
import pandas as pd

# Функция получения метрик
def get_metrics(model):
    metrics = model.val()
    results = metrics.results_dict
    return {
        "mAP50": results["metrics/mAP50(B)"],
        "mAP50-95": results["metrics/mAP50-95(B)"],
        "precision": results["metrics/precision(B)"],
        "recall": results["metrics/recall(B)"],
        "f1": 2 * (results["metrics/precision(B)"] * results["metrics/recall(B)"]) /
              (results["metrics/precision(B)"] + results["metrics/recall(B)"])
    }

if __name__ == '__main__':
    # Эксперимент 1
    model1 = YOLO("yolo11s.pt")
    model1.train(data="./project/data.yaml", epochs=50, imgsz=640, batch=4, name="exp1")
    results1 = get_metrics(model1)

    # Эксперимент 2
    model2 = YOLO("yolo11s.pt")
    model2.train(data="./project/data.yaml", epochs=100, imgsz=640, batch=4, lr0=0.005, lrf=0.0005, name="exp2")
    results2 = get_metrics(model2)

    # Сохранение в DataFrame
    df = pd.DataFrame([results1, results2], index=["Exp1", "Exp2"])
    print(df.round(4))

# Результаты сравнения двух моделей обучения
# =================================================
#        mAP50  mAP50-95  precision  recall      f1
# =================================================
# Exp1  0.9287    0.7899     0.8362  0.9925  0.9077
# Exp2  0.9950    0.8520     0.9467  1.0000  0.9726