use std::sync::{Arc, Mutex};

use autopilot::bitmap;

pub struct NumberViewer {
    number: Option<u32>,
    changed: bool,
    rect: autopilot::geometry::Rect,
    active: bool,
    drop: bool,
}

impl NumberViewer {
    pub fn new(rect: [i32; 4]) -> Self {
        Self {
            number: None,
            changed: false,
            rect: autopilot::geometry::Rect { origin: autopilot::geometry::Point::new(rect[0] as f64, rect[1] as f64), size: autopilot::geometry::Size::new(rect[2] as f64, rect[3] as f64) },
            active: true,
            drop: false,
        }
    }

    pub fn changed(&self) -> bool {
        self.changed
    }

    pub fn number(&self) -> Option<u32> {
        return self.number
    }

    pub fn update(this: &Arc<Mutex<Self>>) {
        if this.lock().unwrap().active {
            let bmp = bitmap::capture_screen_portion(this.lock().unwrap().rect);
            if let Ok(bmp) = bmp {
                let img = bmp.image;

                let tmp_img_path = "/tmp/tmp_img.png"; // Adjust path as needed
                img.save(tmp_img_path).expect("Failed to save temporary image");

                // Use Tesseract to perform OCR
                let mut tess = tesseract::Tesseract::new(None, Some("eng"))
                    .expect("Failed to create Tesseract instance");

                if let Ok(mut tess) = tess.set_image(tmp_img_path) {
                    let mut this = this.lock().unwrap();
                    let old_number: Option<u32> = this.number;
                    this.number = tess.get_text().ok().and_then(|x| x.parse::<u32>().ok());

                    this.changed = old_number != this.number;
                }
            }
        }
    }

    pub fn daemon(this: Arc<Mutex<Self>>) {
        while !this.lock().unwrap().drop {
            Self::update(&this);
            std::thread::sleep(Duration::from_millis(500));
        }
    }
}
