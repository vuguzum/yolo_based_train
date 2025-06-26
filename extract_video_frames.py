import imageio.v2 as iio
import os
import cv2

def extract_frames_imageio(video_path, output_folder, start_time=0, end_time=None):
    os.makedirs(output_folder, exist_ok=True)

    reader = iio.get_reader(video_path, format='ffmpeg')
    fps = reader.get_meta_data()['fps']
    total_frames = len(reader)
    duration = total_frames / fps

    print(f"Видео FPS: {fps}, Общая длительность: {duration:.2f} сек.")

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames

    for i, frame in enumerate(reader):
        if not (start_frame <= i < end_frame):
            continue
        frame_filename = os.path.join(output_folder, f"frame_{i:06d}.jpg")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # если нужен BGR
        cv2.imwrite(frame_filename, frame)

    print(f"Извлечено кадров: {end_frame - start_frame}")

# Вызов
extract_frames_imageio(
    video_path="./SRC_VIDEO/2_1.mov",
    output_folder="./frames",
    start_time=28,
    end_time=29
)