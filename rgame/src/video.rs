use opencv::{core::{Mat, MatExprTraitConst, MatTraitConst, Point, Point2f, Scalar}, highgui, imgproc, videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst}};




pub struct VideoDetector {
    
}
impl VideoDetector {
    pub fn video() -> opencv::Result<()> {
        // Open the default camera
        let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
        if !videoio::VideoCapture::is_opened(&cam)? {
            panic!("Unable to open default camera!");
        }

        // Create a window to display the results
        highgui::named_window("Ball Tracking", highgui::WINDOW_AUTOSIZE)?;

        loop {
            let mut frame = Mat::default();

            // Capture a frame from the camera
            cam.read(&mut frame)?;
            if frame.size()?.width == 0 {
                continue;
            }

            // Convert the frame to HSV color space
            let mut hsv_frame = Mat::default();
            imgproc::cvt_color(&frame, &mut hsv_frame, imgproc::COLOR_BGR2HSV, 0)?;

            // Define the HSV color range for the ball (e.g., for a green ball)
            let lower_bound = Scalar::new(29.0, 86.0, 6.0, 0.0); // Adjust according to the ball's color
            let upper_bound = Scalar::new(64.0, 255.0, 255.0, 0.0);

            // Create a mask for the ball based on the color range
            let mut mask = Mat::default();
            opencv::core::in_range(&hsv_frame, &lower_bound, &upper_bound, &mut mask)?;

            // Perform a series of dilations and erosions to remove small blobs in the mask
            let kernel = Mat::ones(9, 9, opencv::core::CV_8U)?.to_mat()?;
            imgproc::erode(&mask, &mut mask, &kernel, Point::new(-1, -1), 2, opencv::core::BORDER_CONSTANT, Scalar::default())?;
            imgproc::dilate(&mask, &mut mask, &kernel, Point::new(-1, -1), 2, opencv::core::BORDER_CONSTANT, Scalar::default())?;

            // Find contours in the mask
            let mut contours = opencv::types::VectorOfMat::new();
            imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

            // Only proceed if at least one contour was found
            if !contours.is_empty() {
                // Find the largest contour in the mask and use it to compute the minimum enclosing circle and centroid
                let mut largest_contour = contours.get(0)?;
                let mut max_area = imgproc::contour_area(&largest_contour, false)?;

                for i in 1..contours.len() {
                    let contour = contours.get(i)?;
                    let area = imgproc::contour_area(&contour, false)?;
                    if area > max_area {
                        max_area = area;
                        largest_contour = contour;
                    }
                }

                // Calculate the moments of the largest contour
                let moments = imgproc::moments(&largest_contour, false)?;

                if moments.m00 != 0.0 {
                    // Compute the center of the ball (centroid)
                    let c_x = (moments.m10 / moments.m00) as i32;
                    let c_y = (moments.m01 / moments.m00) as i32;

                    // Draw the centroid on the frame
                    imgproc::circle(&mut frame, Point::new(c_x, c_y), 5, Scalar::new(0.0, 0.0, 255.0, 0.0), -1, imgproc::LINE_8, 0)?;

                    // Optionally, compute the minimum enclosing circle around the ball
                    let (mut center, mut radius) = (Point2f::default(), 0.0);
                    imgproc::min_enclosing_circle(&largest_contour, &mut center, &mut radius)?;
                    
                    // Draw the circle on the frame
                    if radius > 10.0 {
                        imgproc::circle(&mut frame, Point::new(center.x as i32, center.y as i32), radius as i32, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
                    }
                }
            }

            // Show the result
            highgui::imshow("Ball Tracking", &frame)?;

            // Break the loop if 'q' is pressed
            if highgui::wait_key(10)? == 113 {
                break;
            }
        }

        Ok(())
    }
}