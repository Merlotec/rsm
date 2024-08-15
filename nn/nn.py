from PIL import Image, ImageGrab, ImageOps, ImageEnhance, ImageChops
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import cv2
from PIL import ImageGrab, ImageTk
import time
import pytesseract
from pynput import mouse
import os
import matplotlib.pyplot as plt
from playsound import playsound
import platform
import threading 
import pyautogui
import logging
import tkinter as tk


await_drawing = False
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial x, y coordinates
ex, ey = -1, -1
active_rect = 1
direction = 1
min_tau = 0.9
g_dis_sign = -1

board = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]

rects = [
    (933, 235, 90, 50), # Ball c
    (1031, 442, 85, 23), # Board green - close view
    (839, 676, 83, 44), # Board green - standby view
    (1511, 196, 89, 65), # Start state detector
    (836, 817, 31, 28), # Outcome number detector
    (934, 740, 691, 170), # Betspace
    (1543, 432, 140, 97), # Secondary board green - close view
]

zero_width = 20.0


def bet_pos(num, w, h, zw) -> tuple[float, float]:
    if num == 0:
        return (num - zw / 2, h / 2)
    
    row = num % 3
    col = np.floor(num / 3)

    bw = w / 12
    bh = h / 12

    cx = bw / 2
    cy = bh / 2

    return (bw * col + cx, bh * row + cy)


def click_at_position(x, y):
    # Move the mouse to the specified position
    pyautogui.moveTo(x, y)
    time.sleep(0.5)
    # Perform the click action
    pyautogui.click()

def place_bets(nums, rect, zw):
    for num in nums:
        x, y, w, h = rect
        px, py = bet_pos(num, w, h, zw)
        px += x
        py += y
        click_at_position(px, py)

def fake_bet(rect, zw):
    x, y, w, h = rect
    px, py = bet_pos(17, w, h, zw) + (x, y)
    click_at_position(px, py)
    time.sleep(0.5)
    click_at_position(px, py) # Undo bet

def average_color(rect):
    x, y, width, height = rect
    # Capture the screen in the specified region
    screen = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    
    # Convert the captured region to a NumPy array
    np_image = np.array(screen)
    
    # Calculate the average color by averaging across the width and height
    avg_color_per_row = np.mean(np_image, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    
    # Convert the average color to integer values
    avg_color = tuple(avg_color.astype(int))
    
    return avg_color

def beep_after_x_seconds(x: float, pr_d = None, tau_b = None):
    def play_beep():
        time.sleep(x)  # Wait for x seconds
        print(f"BEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEP! (pr_d:{pr_d}), (tau_b:{tau_b})")
        if platform.system() == 'Windows':
            # Use the default system beep sound on Windows
            duration = 1000  # milliseconds
            frequency = 1000  # Hertz
            os.system(f'powershell "[console]::beep({frequency}, {duration})"')
        elif platform.system() == 'Darwin':  # macOS
            # Use afplay to play a sound on macOS
            playsound('/System/Library/Sounds/Glass.aiff')
        elif platform.system() == 'Linux':
            # Play a sound using a simple method on Linux
            try:
                os.system('play -q -n synth 0.1 sin 880')
            except:
                # If 'beep' is not installed, use a wav file as fallback
                playsound('/usr/share/sounds/alsa/Front_Center.wav')  # Adjust this path as needed
        else:
            print("Unsupported platform")

    # Spawn a new thread for the beep function
    beep_thread = threading.Thread(target=play_beep)
    beep_thread.start()

def load_or_create_model(model_path):
    if os.path.exists(model_path):
        # Load the existing model
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("Loaded existing model.")
    else:
        # Create a new model
        model = models.Sequential([
            layers.InputLayer(input_shape=(1,)),  # Single input feature
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(1)  # Output a single continuous value
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("Created a new model.")
    
    return model

def load_or_create_fall_model(model_path):
    if os.path.exists(model_path):
        # Load the existing model
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Loaded existing model.")
    else:
        # Create a new model
        model = models.Sequential([
            layers.InputLayer(input_shape=(2,)),  # 2 continuous inputs
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(37, activation='softmax')  # 37 discrete outcomes (probabilities)
        ])
        
        # Compile the model with categorical crossentropy loss and an appropriate optimizer
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print("Created a new model.")
    
    return model

def add_training_data(model, x_train, y_train, model_path, epochs=300, batch_size=16, validation_split=0.2):
    # Train (or further train) the model with the new data
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Save the updated model to the file
    model.save(model_path)
    print(f"Model saved to {model_path}.")


def add_training_fall_data(model, x_train, y_train, model_path, epochs=10, batch_size=16):
    # Train (or further train) the model with the new data
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the updated model to the file
    model.save(model_path)
    print(f"Model saved to {model_path}.")

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Screen", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        r = (ix, iy, x, y)
        rects[active_rect] = r
        print("New Rect {r}")
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Screen", img)

def on_click(x, y, button, pressed):
    global ix, iy, ex, ey, drawing, await_drawing

    if pressed:
        if await_drawing:
            # Start drawing the rectangle
            drawing = True
            ix, iy = x, y
    elif drawing:
        # Finish drawing the rectangle
        drawing = False
        await_drawing = False
        ex, ey = x, y
        w = None
        h = None
        px = None
        py = None
        if ix <= ex:
            px = ix
            w = ex - ix
        else:
            px = ex
            w = ix - ex

        if iy <= ey:
            py = iy
            h = ey - iy
        else:
            py = ey
            h = iy - ey
        r = (px, py, w, h)
        rects[active_rect] = r
        print(f"rect: {r}")

def on_move(x, y):
    if drawing:
        # Update the ending coordinates as the mouse moves
        global ex, ey
        ex, ey = x, y

class Detector:
    previous_frame = None
    previous_time = None
    time_threshold = 0.2
    tau = None

    def __init__(self, rect, time_threshold = 0.25, threshold=30, min_contour_area = 500):
        x, y, width, height = rect
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.time_threshold = time_threshold

    # def __init__(self, x, y, width=50, height=50, threshold=30, min_contour_area=500):
    #     self.x = x
    #     self.y = y
    #     self.width = width
    #     self.height = height
    #     self.threshold = threshold
    #     self.min_contour_area = min_contour_area

    def run_detect_movement_green(self):
        while True:
            if self.detect_movement_green():
                t = time.time()
                if self.previous_time is not None:
                    self.tau = t - self.previous_time
                self.previous_time = t

    def detect_movement_green(self):

        # Capture the screen in the specified region
        screen = ImageGrab.grab(bbox=(self.x, self.y, self.x + self.width, self.y + self.height))
        frame = np.array(screen)
        
        # Convert the captured frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the range for the color green in HSV
        lower_green = np.array([40, 40, 40])  # Lower bound for green
        upper_green = np.array([80, 255, 255])  # Upper bound for green
        
        # Create a mask for green color
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        # Apply the mask to the frame to isolate green regions
        green_frame = cv2.bitwise_and(frame, frame, mask=green_mask)
        
        # Convert the green frame to grayscale
        gray = cv2.cvtColor(green_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Blur to reduce noise
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        # Compute the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        self.previous_frame = gray


        if self.previous_time:
            if time.time() - self.previous_time < self.time_threshold:
                return False

        # Apply a threshold to get a binary image
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Find contours of the movement
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue  # Ignore small movements
            
            # If significant movement is detected
            (x, y, w, h) = cv2.boundingRect(contour)
            print(f"Green object movement detected at region: x={x}, y={y}, width={w}, height={h}")
            self.previous_time = time.time()
            return True
        return False


    def detect_movement(self) -> bool:
        # Capture the screen in the specified region
        screen = ImageGrab.grab(bbox=(self.x, self.y, self.x + self.width, self.y + self.height))
        frame = np.array(screen)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Blur to reduce noise and small changes

        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        # Compute the absolute difference between the current frame and previous frame
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        self.previous_frame = gray

        if self.previous_time:
            if time.time() - self.previous_time < self.time_threshold:
                return False

        # Apply a threshold to get binary image
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Find contours of the movement
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            
            # If we detect significant movement
            (x, y, w, h) = cv2.boundingRect(contour)
            print(f"Movement detected at region: x={x}, y={y}, width={w}, height={h}")
            self.previous_time = time.time()
            return True
        # For visualization (optional, can be removed in production)
        # cv2.imshow("Movement Detection", thresh)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return
        return False

def capture_and_read_text(rect):
    x, y, width, height = rect
    # Capture the screen region
    bbox = (x, y, x + width, y + height)
    screen = ImageGrab.grab(bbox=bbox)

    # Convert the image to RGB
    rgb_image = screen.convert("RGB")

    # Split into individual channels
    r, g, b = rgb_image.split()

    # Increase contrast of the red channel to isolate red text
    enhancer = ImageEnhance.Contrast(r)
    r_enhanced = enhancer.enhance(2.0)

    # Convert red channel to binary image with thresholding
    r_bw = r_enhanced.point(lambda x: 0 if x < 150 else 255, '1')

    # Now handle the gray and white text
    gray = ImageOps.grayscale(screen)
    enhancer = ImageEnhance.Contrast(gray)
    enhanced_gray = enhancer.enhance(2.0)

    # Convert grayscale image to binary
    bw_gray = enhanced_gray.point(lambda x: 0 if x < 150 else 255, '1')

    # Combine the two binary images (red text and gray/white text)
    combined = ImageChops.lighter(r_bw, bw_gray)

    # Optional: Save the processed image for debugging
    combined.save('processed_image.png')

    # Invert the image before OCR
    inverted_image = ImageOps.invert(combined)

    # Optional: Save the processed image for debugging
    inverted_image.save('processed_image_inverted.png')

    # Perform OCR to extract the number
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    number = pytesseract.image_to_string(inverted_image, config=custom_config)

    # Clean the OCR result to remove any non-numeric characters
    number = number.strip()
    return number


# Define the neural network model
def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    # Add a few dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    
    # Output layer with softmax activation for classification
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def simulate(rect_x, rect_b, model, dist_model):
    x_sets = []
    b_sets = []

    prev_x = None
    prev_b = None

    pr_t = None
    pr_d = None

    tau_x = None
    tau_b = None

    t_b = None

    det_x = Detector(rect_x)
    det_b = Detector(rect_b, time_threshold=1.0)

    while True:
        if det_x.detect_movement():
            t = time.time()
            if prev_x and not tau_x:
                tau_x = t - prev_x
                pr_t = model.predict(np.array([[tau_x]]))[0][0]
                pr_d = dist_model.predict(np.array([[tau_x]]))[0][0]
                x_sets.append((t, tau_x))
                beep_after_x_seconds(pr_t.item() - 0.5, (pr_d) % (np.pi * 2.0))
            prev_x = t
        # if det_b.detect_movement_green():
        #     t = time.time()
        #     if prev_b:
        #         tau_b = t - prev_b
        #         b_sets.append((t, tau_b))
        #     prev_b = t
        key = cv2.pollKey() & 0xFF
        if key == ord('v'):
            return (x_sets, b_sets)
        

g_tau = None
g_prev_b = None
kill_green_worker = False
def begin_green_worker(rect_b, rect_b2, dir_sign):
    global g_tau, kill_green_worker, g_prev_b
    det_b = Detector(rect_b, time_threshold=1.0)
    det_b2 = Detector(rect_b2, time_threshold=1.0)

    t_1 = None
    t_2 = None

    while not kill_green_worker:
        if det_b.detect_movement_green():
            t = time.time()
            g_prev_b = t
            if t_2 is not None and (t_1 is None or t_2 > t_1):
                hd = t - t_2
                print("Set tau_g")
                if dir_sign < 0:
                    g_tau = hd * 4.0
                else:
                    g_tau = hd * (4.0 / 3.0)
            t_1 = t
            
        if det_b2.detect_movement_green():
            t = time.time()
            if t_1 is not None and (t_2 is None or t_1 > t_2):
                hd = t - t_1
                print("Set tau_g")
                if dir_sign > 0:
                    g_tau = hd * 4.0
                else:
                    g_tau = hd * (4.0 / 3.0)
            t_2 = t
    kill_green_worker = False
    pass
           


def simulate_imm(rect_x, rect_b, rect_b2, model, dist_model, dir_sign):
    global min_tau, kill_green_worker, g_tau

    x_sets = []
    b_sets = []

    prev_x = None
    prev_b = None

    pr_t = None
    pr_d = None

    tau_x = None
    tau_b = None

    t_b = None

    det_x = Detector(rect_x)

    thread = threading.Thread(target=lambda: begin_green_worker(rect_b, rect_b2, dir_sign))
    thread.start()

    tstart = time.time()

    while True:
        if det_x.detect_movement():
            t = time.time()
            if prev_x:
                old_tau_x = tau_x
                if old_tau_x is None:
                    old_tau_x = 2.0
                tau_x = t - prev_x
                if tau_x > min_tau and tau_x < old_tau_x * 2.0:
                    pr_t = model.predict(np.array([[tau_x]]))[0][0]
                    pr_dist = dist_model.predict(np.array([[tau_x]]))[0][0]
                    pr_d = pr_dist
                    kill_green_worker = True
                    return (pr_t, pr_d, g_tau)
            prev_x = t
        # if det_b.detect_movement_green():
        #     t = time.time()
        #     if prev_b:
        #         tau_b = t - prev_b
        #         b_sets.append((t, tau_b))
        #     prev_b = t
        key = cv2.pollKey() & 0xFF
        if key == ord('x') or time.time() - tstart > 20.0:
            return None

def run_detector(rect_x, model, dist_model):
    datapoints = []

    det_x = Detector(rect_x)
    while True:
        if det_x.detect_movement():
            datapoints.append(time.time())
            if len(datapoints) == 2:
                dt = datapoints[1] - datapoints[0]
                pr0 = model.predict(np.array([[dt]]))[0][0]
                beep_after_x_seconds(pr0.item())
        key = cv2.pollKey() & 0xFF
        if key == ord('x'):
            t = time.time()
            last_t = datapoints[len(datapoints) - 1]
            last_seg = (1 / (last_t - datapoints[len(datapoints) - 2])) * (t - last_t) * np.pi * 2.0
            training_data = []
            prev = None
            i = 0
            for point in datapoints:
                bad = False
                if prev is not None:
                    for j in range(i, len(datapoints), 1):
                        if datapoints[j] < point:
                            bad = True
                            for _, _, tp in training_data:
                                tp += 1
                            break
                    if not bad:
                        delta_t = point - prev
                        rots = len(datapoints) - i - 1
                        dis = rots * np.pi * 2.0 + last_seg
                        dis_pr = dist_model.predict(np.array([[delta_t]]))[0][0]
                        input_array = np.array([[delta_t]]) 
                        pr = model.predict(input_array)
                        training_data.append([delta_t, t - point, dis, pr[0][0], dis_pr])
                prev = point
                i += 1
            
            return training_data
        

def clockwise_distance(angle1, angle2):
    # Normalize the angles to be within [0, 2*pi]
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)

    # Clockwise distance
    clockwise_distance = (angle2 - angle1) % (2 * np.pi)

    return clockwise_distance

def anticlockwise_distance(angle1, angle2):
    # Normalize the angles to be within [0, 2*pi]
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)

    # Anticlockwise distance
    anticlockwise_distance = (angle1 - angle2) % (2 * np.pi)

    return anticlockwise_distance

def pos_to_num(pos):
    return board[pos]

def num_to_pos(num):
    return board.index(num)

def pos_to_dis(pos):
    return pos * 2.0 * np.pi / len(board)

def discretise(v):
    num = round(v / (np.pi * 2.0) * 37)
    if num == 37:
        return 0
    else:
        return num
    
def undiscretise(num):
    return (float(num) / 37.0) * np.pi * 2.0

# dir_sign 1 for clockwise ball, else -1 for anticlockwise ball
def dis_to_fall(dis_f, rot_b, dis_x, dir_sign):
    a_f = dis_f + rot_b * -dir_sign

    a_x = dis_x * dir_sign
    
    if dir_sign > 0:
        return clockwise_distance(a_x, a_f)
    else:
        return anticlockwise_distance(a_x, a_f)
    
def play(probs, n):
    return np.argsort(probs)[-n:][::-1]

def calc_odds(n) -> float:
    return n / 37.0

def primative_resolve(tau_b, loc_d, delta_t, pr_d, dir_sign, take):
    global board
    end = loc_d + (5.9 + (1 / tau_b) * 0.5) * dir_sign

    dist_b = tau_b * delta_t * -dir_sign
    pos = dist_b % np.pi * 2.0

    bpos = end - pos
    pn = bpos % np.pi * 2.0
    n = discretise(pn)
    moves = []

    print(f"Primative predict centered on n: {n}")

    threshold = (take - 1) / 2

    i = 0
    for mv in board:
        if min(np.abs(n - i), np.abs(36 + i - n)) <= threshold:
            moves.append(mv)
        i += 1

    return moves

def resolve_moves(pr_out, tau_b, t, dir_sign):
    dist_b = tau_b * t * -dir_sign
    pos = dist_b % np.pi * 2.0
    moves = []
    for move in pr_out:
        md = undiscretise(move)
        p = dist_b + md * dir_sign
        bpos = p - pos
        pn = bpos % np.pi * 2.0
        n = discretise(pn)
        mv = pos_to_num(n)
        moves.append(mv)
    return moves


def calc_dis_b(delta_t, tau_b):
    tau1 = tau_b + delta_t * 0.02
    v0 = np.pi * 2.0 / tau_b
    v1 = np.pi * 2.0 / tau1
    d = ((v0 + v1 )/ 2) * delta_t
    return d

def track_b_state():
    global g_tau, g_prev_b
    while True:
        if g_tau is not None and g_prev_b is not None:
            loc_b = (calc_dis_b(time.time() - g_prev_b, g_tau) * g_dis_sign) % np.pi * 2.0
            print(f"Tick, board: {loc_b}  (g_tau: {g_tau}, g_prev_b: {g_prev_b})")
            time.sleep(0.5)

def run_logic(model, dist_model, fall_model, dir_sign):
    global rects, zero_width, g_tau, g_prev_b
    det_b = Detector(rects[2], time_threshold=1.0)

    num = None

    prev_b = None
    tau_b = None
    loc_d = None
    sign_d = None
    pr_t = None
    pr_d = None
    pr_b = None

    moves = None

    pr_fall = None

    state = 0
    acc_roi = 1.0
    total_n = 0
    total_roi = 1

    wins = 0

    print("State 0")
    while True:
        if state == 0:
            if det_b.detect_movement_green():
                t = time.time()
                if prev_b is not None:
                    tau_b = t - prev_b
                    g_tau = tau_b
                    print("Board speed set")
                prev_b = t
                g_prev_b = prev_b
            # Wait for zoom.
            nc = np.linalg.norm(average_color(rects[3])) 
            key = cv2.pollKey() & 0xFF
            if key == ord('d'):
                print(f"colmag: {nc}")
            if nc < 90.0:
                # dark, so we know to start simulating.
                print("State 1")
                state = 1
        elif state == 1:
            print("Running sim...")
            if tau_b is None:
                print("No board speed")
                print("State 3")
                state = 3
                continue
            sim = simulate_imm(rects[0], rects[1], rects[6], model, dist_model, dir_sign)
            if sim:
                print(f"Simulation output: {sim}")
                t = time.time()
                (pr_t, pr_d, tau_gb) = sim
                if tau_gb is not None:
                    print("Using taugb")
                    tau_b = tau_gb
                    prev_b = g_prev_b
                pr_b = pr_t * (np.pi * 2.0) / tau_b
                loc_d = (pr_d * dir_sign) % (np.pi * 2.0)
                loc_b = (pr_b)
                beep_after_x_seconds(pr_t.item(), loc_d, loc_b)
                sign_d = loc_d * dir_sign
                pr_fall_out = fall_model.predict(np.array([[tau_b, sign_d]]))[0][0]
                play_out = play(pr_fall_out, 14)
                loc_b = calc_dis_b((t - prev_b) + pr_t, tau_b)
                #moves = resolve_moves(play_out, tau_b, prev_b, dir_sign)
                moves = primative_resolve(tau_b, loc_d, pr_t, pr_d, dir_sign, 17)
                #place_bets(moves, rects[5], zero_width)
                print(f"Playing moves: {moves}")
                print("State 2")
                state = 2
                try:
                    num = int(capture_and_read_text(rects[4]))
                except ValueError:
                    num = None
                print(f"oldnum: {num}")
            else:
                print("Sim failed, state 0")
                state = 0
        elif state == 2:
            try:
                newnum = int(capture_and_read_text(rects[4]))
            except ValueError:
                newnum = None

            if num != newnum and newnum is not None:
                # Complete analysis
                total_n += 1
                print(f"OUTCOME: {newnum}")
                
                if newnum in moves:
                    roi = 36.0 / len(moves)
                    acc_roi += roi
                    wins += 1
                    print(f"WE WIN!!!!! ROI: {roi}, wins: {wins}, winrate: {wins / total_n}, expected: {calc_odds(len(moves))}")
                else:
                    print("WE LOSE :(")

                #total_roi = acc_roi / total_n
                #print(f"Total ROI: {total_roi}")
                pos = num_to_pos(newnum)
                fall = dis_to_fall(pos_to_dis(pos), pr_b, pr_d, dir_sign)
                fall_n = discretise(fall)
                #add_training_fall_data(fall_model, np.array([[tau_b, sign_d]]), np.array([fall_n]), 'fall.h5') # Train the model
                predpr = (5.9 + (1 / tau_b) * 0.5) * dir_sign
                logging.info(f"fall: {fall}, pred: {predpr % np.pi * 2.0}")
                print(f"Training on fall: {fall_n}")
                print("State 0")
                state = 0
        elif state == 3:
            nc = np.linalg.norm(average_color(rects[3])) 
            if nc > 200.0:
                state = 0
                print("State 0")





def plotdata(xs, ys, ps):
    test_input_times = np.linspace(1, 10, 100).reshape(-1, 1)
    predicted_output_times = model.predict(test_input_times)

    # Plot the predictions
    plt.scatter(xs, ys, label='True Data')
    plt.plot(xs, ps, color='red', label='Model Predictions')
    plt.xlabel('Time for One Complete Circle')
    plt.ylabel('Time Until Fall')
    plt.legend()
    plt.show()


# original_image = Image.open("plate.png") 

# root = tk.Tk()
# root.title("Continuous Image Rotation")

# # Create a label to display the image
# image_label = tk.Label(root)
# image_label.pack()

# g_dis_sign = -1

# def update_image():
#     global g_tau, g_prev_x, g_dis_sign
#     if g_prev_x is not None and g_tau is not None:
#         rotation_angle = calc_dis_b(time.time() - g_prev_x, g_tau) * g_dis_sign

#         # Rotate the image by the global angle
#         rotated_image = original_image.rotate(rotation_angle, expand=True)

#         # Convert to ImageTk format
#         tk_image = ImageTk.PhotoImage(rotated_image)

#         # Update the label with the new image
#         image_label.config(image=tk_image)
#         image_label.image = tk_image

#         # # Increment the rotation angle
#         # rotation_angle += 2  # Adjust this value for faster/slower rotation
#         # if rotation_angle >= 360:
#         #     rotation_angle = 0

#     # Schedule the update_image function to run again after 50ms
#     root.after(50, update_image)

if __name__ == "__main__":
# Configure logging

    # Initialize the display with the original image


    logging.basicConfig(
        filename='app.log',          # Log file name
        filemode='a',                # Append mode; use 'w' for overwrite
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO           # Set the minimum logging level
    )

    cv2.namedWindow("Screen")
    
    listener = mouse.Listener(on_click=on_click, on_move=on_move)
    listener.start()

    model = load_or_create_model('model.h5')
    dist_model = load_or_create_model('dist.h5')
    fall_model = load_or_create_fall_model('fall.h5')

    threading.Thread(target=track_b_state).start()

    while True:
        #cv2.imshow("Screen", img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('l'):
            active_rect += 1
            if active_rect > 6:
                active_rect = 0
            print(f"Active rect: {active_rect}")
        elif key == ord('c'):
            if not await_drawing: 
                await_drawing = True
                print("awainting draw")
        elif key == ord('p'):
            x, y, w, h = rects[active_rect]

            # Capture the region
            screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))

            # Display the screenshot
            plt.imshow(screenshot)
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.show()

        elif key == ord('r'):
            run_logic(model, dist_model, fall_model, g_dis_sign)
        elif key == ord('s'):
            x_data, b_data = simulate(rects[0], rects[1], model, dist_model)
            # print(f"xdata: {x_data}")
            # print(f"bdata: {b_data}")
            # tx, xs = zip(*x_data)
            # tb, bs = zip(*b_data)

            # plt.scatter(tx, xs, label='Ball')
            # plt.scatter(tb, bs, label='Board')
            # plt.xlabel('Time for One Complete Circle')
            # plt.ylabel('Delta time')
            # plt.legend()
            # plt.show()
        elif key == ord('t'):
            #start
            if not rects[0]:
                print("Set x rect")
            else:
                training_data = run_detector(rects[0], model, dist_model)
                if training_data:
                    print(f"Training: \n{training_data}")
                    xs, ys, ds, ps, pds = zip(*training_data)
                    plotdata(xs, ys, ps)
                    plotdata(xs, ds, pds)
                    add_training_data(model, np.array(xs), np.array(ys), 'model.h5')
                    add_training_data(dist_model, np.array(xs), np.array(ds), 'dist.h5')


        
    cv2.destroyAllWindows()


