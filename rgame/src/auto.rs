use std::time::SystemTime;

use autopilot::geometry::Point;
use bevy::{math::{Vec2, Quat, Rect, Mat2, Vec3}, render::color::Color};
use spindle::bb::Params;


#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Ring {
    pub centre: Vec2,
    pub squish: f32,
    pub rot: f32,
    pub r: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PixelBoard {
    pub pixels: Vec<(Vec2, f64)>,
    pub ring: Ring,
    pub s: f32,
}

pub struct PixelSim {
    pub clockwise: bool,
    pub t_0: SystemTime,
    pub d_0: f64,
    pub data: Vec<(f64, f64)>, // time, dist
    pub head_vel: Option<f64>,
    pub board: PixelBoard,
}

impl PixelSim {
    pub fn start(board: PixelBoard, d_0: f64, clockwise: bool) -> Self {
        Self {
            clockwise,
            t_0: SystemTime::now(),
            d_0,
            data: vec![(0.0, 0.0)],
            head_vel: None,
            board,
        }
    }

    pub fn dir_mul(&self) -> f64 {
        if self.clockwise {
            1.
        } else {
            -1.
        }
    }

    pub fn update(&mut self) -> image::ImageResult<()> {    
        let t = SystemTime::now().duration_since(t_0).unwrap().as_secs_f64();

        let last_d = self.data.last().unwrap().1;

        if let Some(pos) = self.board.calculate_leading_pos(target_color, threshold, clockwise)? {
            let last_d = self.data.last().1;
            let last_p = spindle::game::dis_to_pos(self.d_0 + last_d * self.dir_mul());

            let delta_d = directional_dis(last_p, pos, self.clockwise);

            let new_d = last_d + delta_d;

            self.data.push((t, new_d));
        }

        Ok(())
    }

    pub fn estimate_params(&self) -> Option<Params> {
        spindle::bb::solve(ts, ps)
    }

    pub fn velocity(&self) -> f64 {
        
    }
}

impl PixelBoard {
    pub fn from_ring(ring: Ring, s: f32) -> Self {
        
        let mut pixels = Vec::new();

        let rect = ring.rect();
        let l = (rect.width() / s) as usize;
        let h = (rect.height() / s) as usize;
        for x in 0..l {
            for y in 0..h {
                let p = rect.min + Vec2::new(x as f32, y as f32);
                if let Some(pos) = ring.include_pixel(p, s) {
                    pixels.push((p, pos as f64));
                }
            }
        }

        Self { ring, pixels, s }
    }

    pub fn search(&self, target_color: Color, threshold: f32) -> image::ImageResult<Vec<f64>> {
        let rect = self.ring.rect();
        let aurect = autopilot::geometry::Rect::new(
            autopilot::geometry::Point { x: rect.min.x as f64, y: rect.min.y as f64 }, 
            autopilot::geometry::Size::new(rect.width() as f64, rect.height() as f64));

        let frame = autopilot::bitmap::capture_screen_portion(aurect)?;

        let mut output = Vec::new();

        for (px, pos) in self.pixels.iter() {
            let rel_px = *px - self.ring.rect().min;
            let col = frame.get_pixel(Point { x: rel_px.x as f64, y: rel_px.y as f64 });
            let col_c = Color::rgb_u8(col[0], col[1], col[2]);
            let col_v = Vec3::new(col_c.r(), col_c.g(), col_c.b());

            let col_t = Vec3::new(target_color.r(), target_color.g(), target_color.b());
            if col_t.distance(col_v).abs() < threshold {
                output.push(*pos);
            }
        }
        Ok(output)
    }

    pub fn calculate_leading_pos(&self, target_color: Color, threshold: f32, clockwise: bool) -> image::ImageResult<Option<f64>> {
        let mut pixels = self.search(target_color, threshold)?;
        if pixels.len() > 0 {
            if pixels.len() > 1 {
                let avg: f64 = pixels.iter().enumerate().fold(*pixels.first().unwrap(), |a, (i, p)| a + angle_dis(a, *p) / (i + 1) as f64);
                pixels.retain(|p| (p - avg).abs() < std::f64::consts::FRAC_PI_2);
                let min = *pixels.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let max = *pixels.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

                if clockwise {
                    if angle_dis(min, max).is_sign_positive() {
                        return Ok(Some(max));
                    } else {
                        return Ok(Some(min));
                    }
                } else {
                    if angle_dis(min, max).is_sign_positive() {
                        return Ok(Some(min));
                    } else {
                        return Ok(Some(max));
                    }
                }
            } else {

            }
            
        }
        Ok(None)
    }

    fn calculate_real_pos(&self, target_color: Color, threshold: f32, clockwise: bool, h: f64) -> image::ImageResult<Option<f64>> {
        if let Some(p) = self.calculate_leading_pos(target_color, threshold, clockwise)? {
            if clockwise {
                if 0.0 < p && p < std::f64::consts::PI {
                    Ok(Some(p))
                } else {
                    Ok(Some(p - self.lead_for_angle(p, h)))
                    
                }
            } else {
                if 0.0 < p && p < std::f64::consts::PI {
                    Ok(Some(p + self.lead_for_angle(p, h)))
                } else {
                    Ok(Some(p))
                }
            }
        } else {
            Ok(None)
        }
    }

    fn lead_for_angle(&self, pos: f64, h: f64) -> f64 {
        (h * pos.sin() / self.ring.r as f64).tan()
    }
}



fn angle_dis(t1: f64, t2: f64) -> f64 {
    let t1 = t1 % (std::f64::consts::PI * 2.0);
    let t2 = t2 % (std::f64::consts::PI * 2.0);

    let d1 = t2 - t1;
    let d2 = t2 - (t1 + std::f64::consts::PI * 2.0);

    if d2.abs() < d1.abs() {
        d2
    } else {
        d1
    }
}


fn directional_dis(p1: f64, p2: f64, clockwise: bool) -> f64 {
    if clockwise {
        if p2 >= p1 {
            p2 - p1
        } else {
            (std::f64::consts::PI * 2.) + (t2 - t1)
        }
    } else {
        if p2 <= p1 {
            p1 - p2
        } else {
            (std::f64::consts::PI * 2.) + (t1 - t2)
        }
    }
}

impl Ring {
    pub fn rect(&self) -> Rect {
        Rect::new(self.centre.x - self.r, self.centre.y - self.r * self.squish, self.centre.x + self.r, self.centre.y + self.r * self.squish)
    }

    pub fn include_pixel(&self, pixel: Vec2, s: f32) -> Option<f32> {
        let rel_pos = pixel - self.centre;
        let pflat = Mat2::from_angle(self.rot) * rel_pos;

        let corners: [Vec2; 4] = [
            pixel + Vec2::new(s / 2.0, s / 2.0), 
            pixel - Vec2::new(s / 2.0, s / 2.0),
            pixel + Vec2::new(s / 2.0, -s / 2.0), 
            pixel - Vec2::new(s / 2.0, -s / 2.0)
        ];

        let cpos = Mat2::from_angle(-self.rot) * pixel;

        let rot_corners = corners.map(|x| (Mat2::from_angle(-self.rot) * x));

        let mut lt = false;
        let mut mt = false;

        for c in rot_corners {
            let cp = Vec2::new(c.x, c.y / self.squish);
            let len = cpos.length();

            if len < self.r {
                lt = true;
            } else if len > self.r {
                mt = true;
            } else if len == self.r {
                lt = true;
                mt = true;
            }
        }

        if lt && mt {
            let npos = Vec2::new(cpos.x, cpos.y / self.squish);
            let mut theta = npos.angle_between(Vec2::new(0.0, 1.0));
            if theta < 0.0 {
                theta = std::f32::consts::PI * 2.0 - theta;
            }
            Some(theta)
        } else {
            None
        }
    }
}