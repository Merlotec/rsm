use opencv::{
    boxed_ref::BoxedRef, core::{Mat, Mat_AUTO_STEP, Scalar, Size, BORDER_DEFAULT}, imgproc::{self, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY, COLOR_BGR2HSV, RETR_EXTERNAL, THRESH_BINARY}, prelude::*, types::VectorOfMat, videoio::VideoCapture
};
use scrap::{Capturer, Display};
use std::{ops::{Deref, DerefMut}, sync::{atomic::AtomicBool, Arc, Mutex}, time::{Duration, Instant, SystemTime}};
use std::thread;
use std::io::ErrorKind;

pub struct MotionDetector {
    previous_frame: Option<Mat>,
    previous_time: Option<Instant>,
    //time_threshold: Duration,
    use_color: bool,
    color_lower: Scalar,
    color_upper: Scalar,
    threshold: f64,
    min_contour_area: f64,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

impl MotionDetector {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            previous_frame: None,
            previous_time: None,
            //ime_threshold: Duration::from_nanos(2),
            threshold: 35.0,
            min_contour_area: 500.0,
            x,
            y,
            width,
            height,
            color_lower: Scalar::new(40.0, 40.0, 40.0, 0.0),
            color_upper: Scalar::new(80.0, 255.0, 255.0, 0.0),
            use_color: false,
        }
    }

    pub fn new_green(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            previous_frame: None,
            previous_time: None,
            //ime_threshold: Duration::from_nanos(2),
            threshold: 35.0,
            min_contour_area: 500.0,
            x,
            y,
            width,
            height,
            color_lower: Scalar::new(40.0, 40.0, 40.0, 0.0),
            color_upper: Scalar::new(80.0, 255.0, 255.0, 0.0),
            use_color: true,
        }
    }

    pub fn detect(&mut self) -> bool {
        if self.use_color {
            self.detect_motion_color()
        } else {
            self.detect_motion()
        }
    }

    pub fn detect_motion(&mut self) -> bool {
        // let mut cap = Capturer::new(Display::primary().expect("Failed to get primary display"))
        //     .expect("Failed to begin capture");
        // let width = cap.width();
        // let height = cap.height();

        let portion = autopilot::bitmap::capture_screen_portion(autopilot::geometry::Rect { origin: autopilot::geometry::Point::new(self.x as f64, self.y as f64), size: autopilot::geometry::Size::new(self.width as f64, self.height as f64) });

        if portion.is_err() {
            return false;
        }

        // let one_frame = cap
        //     .frame()
        //     .expect("Failed to capture frame");
    
        // Convert the frame to a Mat (2D)
        let portion = portion.unwrap();

        let raw = portion.image.as_rgb8();
        if raw.is_none() {
            println!("Could not convert to rgb8");
            return false;
        }

        let mat = Mat::from_slice(&raw.unwrap()).unwrap();
    
        // Reshape the flat buffer into a 2D matrix (with 4 channels for RGBA)
        let mat = mat.reshape(3, portion.size.height as i32).unwrap();

        // let mat = unsafe { Mat::new_rows_cols_with_data_unsafe(
        //     height as i32,
        //     width as i32,
        //     opencv::core::CV_8UC3, // 8-bit unsigned, 3 channels (RGB)
        //     raw.unwrap().as_ptr() as *mut std::ffi::c_void,
        //     Mat_AUTO_STEP,
        // )
        // .expect("Failed to create OpenCV Mat from raw bytes")
        // };

        // Convert to grayscale
        let mut gray = Mat::default();

        if let Err(_) = imgproc::cvt_color(&mat, &mut gray, COLOR_BGR2GRAY, 0) {
            return false;
        }
        let mut output = Mat::default();
        // Apply GaussianBlur
        imgproc::gaussian_blur(
            &gray,
            &mut output,
            Size::new(21, 21),
            0.0,
            0.0,
            BORDER_DEFAULT,
        )
        .unwrap();

        if self.previous_frame.is_none() {
            self.previous_frame = Some(output);
            return false;
        }

        // Compute the absolute difference between the current frame and the previous frame
        let mut frame_diff = Mat::default();
        opencv::core::absdiff(&self.previous_frame.as_ref().unwrap(), &output, &mut frame_diff)
            .unwrap();
        self.previous_frame = Some(output);

        // if let Some(previous_time) = self.previous_time {
        //     if Instant::now().duration_since(previous_time) < self.time_threshold {
        //         return false;
        //     }
        // }

        // Apply a threshold to get binary image
        let mut thresh = Mat::default();
        imgproc::threshold(
            &frame_diff,
            &mut thresh,
            self.threshold,
            255.0,
            THRESH_BINARY,
        )
        .unwrap();

        // Find contours of the movement
        let mut contours = VectorOfMat::new();
        imgproc::find_contours(
            &thresh,
            &mut contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )
        .unwrap();

        for contour in contours.iter() {
            if imgproc::contour_area(&contour, false).unwrap() < self.min_contour_area {
                continue;
            }

            // If we detect significant movement
            let rect = imgproc::bounding_rect(&contour).unwrap();
            // println!(
            //     "Movement detected at region: x={}, y={}, width={}, height={}",
            //     rect.x, rect.y, rect.width, rect.height
            // );
            self.previous_time = Some(Instant::now());
            return true;
        }

        false
    }

    pub fn detect_motion_color(&mut self) -> bool {
        let portion = autopilot::bitmap::capture_screen_portion(autopilot::geometry::Rect { origin: autopilot::geometry::Point::new(self.x as f64, self.y as f64), size: autopilot::geometry::Size::new(self.width as f64, self.height as f64) });

        if portion.is_err() {
            return false;
        }

        // let one_frame = cap
        //     .frame()
        //     .expect("Failed to capture frame");
    
        // Convert the frame to a Mat (2D)
        let portion = portion.unwrap();

        let raw = portion.image.as_rgb8();
        if raw.is_none() {
            println!("Could not convert to rgb8");
            return false;
        }

        let mat = Mat::from_slice(&raw.unwrap()).unwrap();
    
        // Reshape the flat buffer into a 2D matrix (with 4 channels for RGBA)
        let frame = mat.reshape(3, portion.size.height as i32).unwrap();


        // Convert the captured frame to HSV color space
        let mut hsv_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut hsv_frame, COLOR_BGR2HSV, 0).unwrap();

        // Create a mask for green color
        let mut green_mask = Mat::default();
        opencv::core::in_range(&hsv_frame, &self.color_lower, &self.color_upper, &mut green_mask).unwrap();

        // Apply the mask to the frame to isolate green regions
        let mut green_frame = Mat::default();
        opencv::core::bitwise_and(&frame, &frame, &mut green_frame, &green_mask).unwrap();

        // Convert the green frame to grayscale
        let mut gray = Mat::default();

        if let Err(e) = imgproc::cvt_color(&green_frame, &mut gray, COLOR_BGR2GRAY, 0) {
            println!("imgproc error! {}", e);
            return false;
        }
        let mut output = Mat::default();
        // Apply GaussianBlur
        imgproc::gaussian_blur(
            &gray,
            &mut output,
            Size::new(21, 21),
            0.0,
            0.0,
            BORDER_DEFAULT,
        )
        .unwrap();

        // If the previous frame is None, initialize it and return false
        if self.previous_frame.is_none() {
            self.previous_frame = Some(output);
            return false;
        }

        // Compute the absolute difference between the current frame and the previous frame
        let mut frame_diff = Mat::default();
        opencv::core::absdiff(&self.previous_frame.as_ref().unwrap(), &output, &mut frame_diff).unwrap();
        self.previous_frame = Some(output);


        // Apply a threshold to get a binary image
        let mut thresh = Mat::default();
        imgproc::threshold(&frame_diff, &mut thresh, self.threshold, 255.0, THRESH_BINARY).unwrap();

        // Find contours of the movement
        let mut contours = VectorOfMat::new();
        imgproc::find_contours(&thresh, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, opencv::core::Point::new(0, 0)).unwrap();

        // Process the contours to detect significant movement
        for contour in contours.iter() {
            if imgproc::contour_area(&contour, false).unwrap() < self.min_contour_area {
                continue; // Ignore small movements
            }

            // If significant movement is detected, get the bounding box
            let rect = imgproc::bounding_rect(&contour).unwrap();
            self.previous_time = Some(Instant::now());
            return true;
        }

        false
    }
}


pub struct SmartTrigger {
    detector: Mutex<MotionDetector>,
    trigger: Mutex<Option<SystemTime>>,
    last_flush: Mutex<Option<SystemTime>>,
}


impl SmartTrigger {
    pub fn new(detector: MotionDetector) -> Self {
        Self {
            detector: Mutex::new(detector),
            trigger: Mutex::new(None),
            last_flush: Mutex::new(None),
        }
    }

    // Uses mutexes and atomic operations to uupdate.
    pub fn update(&self) {
        if self.detector.lock().unwrap().detect() {
            let mut tr = self.trigger.lock().unwrap();
            *tr.deref_mut() = Some(SystemTime::now());
        }
    }

    pub fn flush_detect(&self, buffer: Duration) -> Option<SystemTime> {
        let mut tr = self.trigger.lock().unwrap();
        if let Some(trigger) = tr.deref().clone() {
            *tr.deref_mut() = None;
            if self.last_flush.lock().unwrap().map(|x| x + buffer < trigger).unwrap_or(true) {
                let mut lt = self.last_flush.lock().unwrap();
                *lt.deref_mut() = Some(trigger);
                Some(trigger)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn clear(&self) {
        self.trigger.lock().unwrap().take();
        self.last_flush.lock().unwrap().take();
    }
    
    pub fn daemon(this: Arc<Self>) {
        loop {
            Self::update(&this);
            //std::thread::sleep(Duration::from_nanos(50));
        }
    }
}