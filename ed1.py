import cv2
import numpy as np
# закомментированный код нужен для сохранения видео с детекцией
# Захватываем видео с веб-камеры или видеофайла
cap = cv2.VideoCapture(0)  #  '0' для камеры или указать путь к видеофайлу

# Параметры для навигации
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center = frame_width // 2  # Центр кадра
turn_threshold = 20  # Порог для обычного отклонения
sharp_turn_threshold = 100  # Порог для поворота

# Параметры для сохранения видео
# output_filename = 'output_with_turn_detection.avi'
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    height, width = blurred.shape
    roi = blurred[int(height * 0.5):, :]  # ROI - нижняя часть изображения

    # Находим контуры
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_center_x, line_center_y = None, None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                line_center_x = int(M["m10"] / M["m00"])
                line_center_y = int(M["m01"] / M["m00"]) + int(height * 0.5)
                cv2.circle(frame, (line_center_x, line_center_y), 5, (0, 0, 255), -1)
                break

    # Проверка на отклонение и поворот
    if line_center_x is not None:
        deviation = line_center_x - frame_center
        if abs(deviation) < turn_threshold:
            command = "Forward"
        elif deviation > 0:
            command = "Turn right"
        else:
            command = "Turn left"

        # Проверка на резкий поворот
        if abs(deviation) > sharp_turn_threshold:
            turn_message = "Turn Detected"
            cv2.putText(frame, turn_message, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Отображаем команду и координаты точки
        cv2.putText(frame, f"Coords: ({line_center_x}, {line_center_y})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Command: {command}, Deviation: {deviation}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Записываем кадр и показываем
    # out.write(frame)
    resized_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
    cv2.imshow('Robot Navigation', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
