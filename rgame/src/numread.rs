use std::{sync::{Arc, Mutex}, time::Duration};

use autopilot::bitmap;
use bevy::{math::Vec3, prelude::Color};
use image::{imageops::{contrast, grayscale, invert}, ImageBuffer, Luma, Rgb, Rgba};
use opencv::{core::{Mat, MatTraitConst}, imgproc};
use tesseract::Tesseract;
#[derive(Debug)]
pub struct NumberViewer {
    number: Option<u32>,
    changed: bool,
    rect: autopilot::geometry::Rect,
    active: bool,
    pub drop: bool,
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

    // pub fn update(this: &Arc<Mutex<Self>>) {
    //     if this.lock().unwrap().active {
    //         let bmp = bitmap::capture_screen_portion(this.lock().unwrap().rect);
    //         if let Ok(bmp) = bmp {
    //             let img = bmp.image;

    //             let tmp_img_path = "tmp_img.png"; // Adjust path as needed
    //             img.save(tmp_img_path).expect("Failed to save temporary image");

    //             // Use Tesseract to perform OCR
    //             let mut tess = tesseract::Tesseract::new(None, Some("eng"))
    //                 .expect("Failed to create Tesseract instance");

    //             if let Ok(mut tess) = tess.set_image(tmp_img_path) {
    //                 let mut this = this.lock().unwrap();
    //                 let old_number: Option<u32> = this.number;
    //                 this.number = tess.get_text().ok().and_then(|x| x.parse::<u32>().ok());

    //                 this.changed = old_number != this.number;
    //             } else {
    //                 println!("Failed to set image!!!");
    //             }
    //         }
    //     }
    // }



    pub fn update(this: &Arc<Mutex<Self>>) {
        let mut this = this.lock().unwrap();
        if this.active {
            let new_number = Self::read_string(this.rect).and_then(|x| x.parse::<u32>().ok());
            let old_number: Option<u32> = this.number;
            if let Some(old_number) = old_number {
                if let Some(new_number) = new_number {
                    if new_number != old_number {
                        this.changed = true;
                        this.number = Some(new_number);
                    }
                }
            } else {  
                if new_number.is_some() {
                    this.number = new_number;
                }
            }
            

            
        }
    }

    pub fn read_string(rect: autopilot::geometry::Rect) -> Option<String> {
        
        let portion = bitmap::capture_screen_portion(rect).ok()?;

        let rgb_image = portion.image.to_rgb();

        // Split the image into individual R, G, and B channels
        let mut binary = ImageBuffer::new(rgb_image.width(), rgb_image.height());
        for (x, y, pixel) in rgb_image.enumerate_pixels() {
            let col = Color::rgb_u8(pixel.0[0], pixel.0[1], pixel.0[2]);

            if col.r() > 0.7 || col.g() > 0.7 {
                // put white bg pixel
                binary.put_pixel(x, y, Rgb([0, 0, 0]));
            } else {
                binary.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }

        // Save the inverted image for debugging (optional)
        binary.save("processed_img.png").unwrap();
        
        // Use Tesseract to perform OCR
        let mut tess = Tesseract::new(None, Some("eng")).expect("Failed to create Tesseract instance")
            .set_image("processed_img.png").expect("Failed to set image")
            .set_variable("tessedit_char_whitelist", "0123456789").expect("Failed to set whitelist");
        
        tess.set_page_seg_mode(tesseract::PageSegMode::PsmSingleBlock);
    
        let ocr_result = tess.get_text().expect("Failed to extract text");
        // Clean up the OCR result by trimming and returning only digits
        Some(ocr_result.trim().to_string())
    }

    pub fn daemon(this: Arc<Mutex<Self>>) {
        while !this.lock().unwrap().drop {
            Self::update(&this);
            std::thread::sleep(Duration::from_millis(500));
        }
        println!("Ending numread daemon");
    }
}
