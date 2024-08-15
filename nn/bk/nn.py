import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import cv2
from PIL import ImageGrab
import time
import pytesseract
from pynput import mouse
import os
import matplotlib.pyplot as plt
from playsound import playsound
import platform
import threading

await_drawing = False
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial x, y coordinates
ex, ey = -1, -1
active_rect = 1
direction = 1

board = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]

rects = [
    (933, 235, 90, 50), # Ball 
    (1031, 442, 85, 23), # Board green - close view
    None, # Board green - standby view
    None, # Start state detector
    None # Outcome number detector
]

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

def beep_after_x_seconds(x: float, pr_d = None):
    def play_beep():
        time.sleep(x)  # Wait for x seconds
        print(f"BEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEP! ({pr_d})")
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

def add_training_data(model, x_train, y_train, model_path, epochs=300, batch_size=16, validation_split=0.2):
    # Train (or further train) the model with the new data
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

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
    
    # Convert the captured image to text
    text = pytesseract.image_to_string(screen)
    
    return text



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
        
def simulate_imm(rect_x, rect_b, model, dist_model):
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

    tstart = time.time()

    while True:
        if det_x.detect_movement():
            t = time.time()
            if prev_x and not tau_x:
                tau_x = t - prev_x
                pr_t = model.predict(np.array([[tau_x]]))[0][0] - 0.5
                pr_dist = dist_model.predict(np.array([[tau_x]]))[0][0]
                pr_d = (pr_dist) % (np.pi * 2.0)
                x_sets.append((t, tau_x))
                return (pr_t, pr_d)
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
    angle1 = angle1 % (2 * math.pi)
    angle2 = angle2 % (2 * math.pi)

    # Clockwise distance
    clockwise_distance = (angle2 - angle1) % (2 * math.pi)

    return clockwise_distance

def anticlockwise_distance(angle1, angle2):
    # Normalize the angles to be within [0, 2*pi]
    angle1 = angle1 % (2 * math.pi)
    angle2 = angle2 % (2 * math.pi)

    # Anticlockwise distance
    anticlockwise_distance = (angle1 - angle2) % (2 * math.pi)

    return anticlockwise_distance

def num_to_pos(num):
    board.index(num)

def pos_to_dis(pos):
    pos * 2.0 * np.pi / len(board)


# dir_sign 1 for clockwise ball, else -1 for anticlockwise ball
def dis_to_fall(dis_f, rot_b, dis_x, dir_sign):
    a_f = dis_f + rot_b * -dir_sign

    a_x = dis_x * dir_sign
    
    if dir_sign > 0:
        return clockwise_distance(a_x, a_f)
    else:
        return anticlockwise_distance(a_x, a_f)
    


def run_logic(model, dist_model):
    det_b = Detector(rects[2], time_threshold=1.0)

    num = None

    prev_b = None
    tau_b = None

    state = 0

    while True:
        if state is 0:
            if det_b.detect_movement_green():
                t = time.time()
                if prev_b:
                    tau_b = t - prev_b
                prev_b = t
            # Wait for zoom.
            if np.abs(average_color(rects[3])) < 0.3:
                # dark, so we know to start simulating.
                state = 1
        elif state is 1:
            sim = simulate(rects[0], rects[1], model, dist_model)
            if sim:
                (pr_t, pr_d) = sim
                beep_after_x_seconds(pr_t.item() - 0.5, (pr_d) % (np.pi * 2.0))
            state = 2
            num = capture_and_read_text(rects[4])
        elif state is 2:
            newnum = capture_and_read_text(rects[4])
            if num != newnum:
                print(f"Outcome was: {newnum}")



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

if __name__ == "__main__":

    # Create a window and set the mouse callback
    cv2.namedWindow("Screen")
    
    listener = mouse.Listener(on_click=on_click, on_move=on_move)
    listener.start()

    model = load_or_create_model('model.h5')
    dist_model = load_or_create_model('dist.h5')

    while True:
        #cv2.imshow("Screen", img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            if not await_drawing: 
                await_drawing = True
                print("awainting draw")
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