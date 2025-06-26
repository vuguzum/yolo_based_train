from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # Загрузка обученной модели
    model = YOLO('runs/detect/train/weights/best.pt')  # или yolov8n.pt, yolo11n.pt

    # Путь к видеофайлу
    video_path = "./SRC_VIDEO/2_1.MOV"
    cap = cv2.VideoCapture(video_path)

    # Получаем параметры исходного видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Оригинальное разрешение: {width}x{height}, FPS: {fps}")

    # Для сохранения обработанного видео (опционально)
    out = cv2.VideoWriter('./SRC_VIDEO/result_2_1.MP4',
                          #cv2.VideoWriter_fourcc(*'mp4v'),
                          cv2.VideoWriter_fourcc(*'XVID'),                          
                          fps,
                          (height, width))  # после поворота ширина и высота меняются местами

    # Создаем окно и перемещаем его в левый верхний угол экрана
    window_name = 'YOLO Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, 0)

    # Обработка кадров
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Повернуть кадр на 90 градусов по часовой стрелке (портрет -> альбом)
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 2. Детекция
        results = model(frame_rotated)

        # 3. Отрисовка результатов
        annotated_frame = results[0].plot()

        # 4. Масштабирование под высоту 900px (сохраняя пропорции)
        target_height = 900
        scale_factor = target_height / annotated_frame.shape[0]
        new_width = int(annotated_frame.shape[1] * scale_factor)
        resized_frame = cv2.resize(annotated_frame, (new_width, target_height))

        # 5. Сохранение кадра (опционально)
        out.write(annotated_frame)

        # 6. Отображение результата (масштабированное окно)
        cv2.imshow(window_name, resized_frame)

        # Нажми 'q', чтобы выйти досрочно
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()