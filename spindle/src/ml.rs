use std::{io, ops::Range, path::Path, time::SystemTime};

use splines::{Key, Spline};

pub type Series = Vec<(f64, f64)>;
pub type SeriesRef<'a> = &'a [(f64, f64)];

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MlTrainer {
    pub datasets: Vec<Dataset>,
    /// The speed at which all datasets are aligned to.
    pub v_star: f64,
}

impl Default for MlTrainer {
    fn default() -> Self {
        Self::new(-6.0)
    }
}

pub struct Prediction {
    pub head: Series,
    pub path: Dataset,
    pub t_head: f64,
    pub t_end: f64,
    pub x_end: f64,
}

impl Prediction {
    pub fn zero_dis(&self, t: f64) -> Option<f64> {
        Some(self.path.sample(t)? - self.path.sample(self.t_head)?)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Dataset {
    pub series: Series,
    pub spline: Spline<f64, f64>,
    pub offset_t: f64,
    pub offset_x: f64,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MlError {
    NoAlignment,
    InsufficientData,
    SplineError,
}

impl std::fmt::Display for MlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAlignment => f.write_str("NoAlignment"),
            Self::InsufficientData => f.write_str("InsufficientData"),
            Self::SplineError => f.write_str("SplineError"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Alignment {
    pub t: f64,
    pub x: f64,
    pub err: f64,
}

impl Dataset {

    pub fn new_aligned(series: Series, v_star: f64) -> Result<Self, MlError> {
        let (t_star, x_star) = Self::search_v(&series, v_star)?;
        let shifted_series = Self::apply_offset(series, -t_star, -x_star);
        let spline = Self::create_monotonic_spline(&shifted_series).unwrap();


        Ok(Dataset { series: shifted_series, offset_t: -t_star, offset_x: -x_star, spline })
    }
    pub fn new_aligned_spline(series: Series, v_star: f64) -> Result<Self, MlError> {
        let spline = Self::create_monotonic_spline(&series).unwrap();
        let (t_star, x_star) = Self::search_spline(&spline, v_star, 0.001, 1.0, 1000)?;
        let shifted_series = Self::apply_offset(series, -t_star, -x_star);
        let spline = Self::create_monotonic_spline(&shifted_series).unwrap();

        Ok(Dataset { series: shifted_series, offset_t: -t_star, offset_x: -x_star, spline })
    }

    pub fn new_aligned_to(series: Series, other: &Dataset, v_star: f64) -> Result<Self, MlError> {
        let ds = Self::new_aligned(series, v_star)?;
        let head = ds.head(0.0).ok_or(MlError::InsufficientData)?;

        let al = other.align(&head.series, 0.0, 0.5, 60).ok_or(MlError::NoAlignment)?;
        
        ds.realign(al).ok_or(MlError::InsufficientData)
    }

    pub fn realign(self, al: Alignment) -> Option<Self> {
        let series: Series = self.series.into_iter().map(|(t, x)| (t + al.t, x + al.x)).collect();
        Some(Self {
            offset_t: self.offset_t + al.t,
            offset_x: self.offset_x + al.x,
            spline: Self::create_monotonic_spline(&series)?,
            series,
        })
    }

    pub fn extrapolate_linear(mut self, take: usize, delta_t: f64, n: usize) -> Result<Self, MlError> {
        let mut m = 0.0;
        let mut n = 0;
        for i in self.series.len() - take..self.series.len() - 1 {
            m += (self.series[i+1].1 - self.series[i].1)/(self.series[i+1].0 - self.series[i].0);
            n += 1;
        }

        let m_star = m / n as f64;

        let mut t = self.series.last().ok_or(MlError::InsufficientData)?.0;
        let mut x = self.series.last().ok_or(MlError::InsufficientData)?.1;

        for _ in 0..n {
            t += delta_t;
            x += m_star * delta_t;
            self.series.push((t, x));
        }

        self.spline = Self::create_monotonic_spline(&self.series).ok_or(MlError::SplineError)?;

        Ok(self)
    }

    pub fn extrapolate_order(mut self, order: usize, take: usize, delta_t: f64, n: usize) -> Result<Self, MlError> {
        assert!(order > 0);
        let mut diffs: Vec<Vec<(f64, f64)>> = vec![vec![]; order];

        for i in self.series.len() - take..self.series.len() - 1 {
            let t = (self.series[i+1].0 + self.series[i].0) / 2.0;
            let m = (self.series[i+1].1 - self.series[i].1)/(self.series[i+1].0 - self.series[i].0);
            diffs[0].push((t, m));
        }

        for o in 1..order {
            if diffs[o-1].len() >= 2 {
            for i in 0..diffs[o-1].len() - 1 {
                    let t = (diffs[o-1][i+1].0 + diffs[o-1][i].0) / 2.0;
                    let m = (diffs[o-1][i+1].1 - diffs[o-1][i].1)/(diffs[o-1][i+1].0 - diffs[o-1][i].0);
                    diffs[o].push((t, m));
                }
            }
        }

        let mut t = self.series.last().ok_or(MlError::InsufficientData)?.0;
        let mut x = self.series.last().ok_or(MlError::InsufficientData)?.1;

        let mut dstate: Vec<f64> = diffs.iter().filter_map(|x| x.last().map(|v| v.1)).collect();

        for _ in 0..n {
            t += delta_t;
            for o in dstate.len() - 1..0 {
                dstate[o-1] += delta_t * dstate[o];
            }
            x += dstate[0] * delta_t; // First derivative.
            self.series.push((t, x));
        }

        self.spline = Self::create_monotonic_spline(&self.series).ok_or(MlError::SplineError)?;

        Ok(self)
    }

    pub fn extrapolate_higher_deriv(mut self, mut higer_derivatives: Vec<f64>, take: usize, delta_t: f64, n: usize) -> Result<Self, MlError> {

        let mut m = 0.0;
        let mut n = 0;
        for i in self.series.len() - take..self.series.len() - 1 {
            m += (self.series[i+1].1 - self.series[i].1)/(self.series[i+1].0 - self.series[i].0);
            n += 1;
        }

        let m_star = m / n as f64;

        let mut t = self.series.last().ok_or(MlError::InsufficientData)?.0;
        let mut x = self.series.last().ok_or(MlError::InsufficientData)?.1;

        higer_derivatives.insert(0, m_star);

        let mut dstate: Vec<f64> = higer_derivatives;

        for _ in 0..n {
            t += delta_t;
            for o in 0..dstate.len() - 1 {
                let o = dstate.len() - o - 1;
                dstate[o-1] += delta_t * dstate[o];
            }
            x += dstate[0] * delta_t; // First derivative.
            self.series.push((t, x));
        }

        self.spline = Self::create_monotonic_spline(&self.series).ok_or(MlError::SplineError)?;

        Ok(self)
    }

    pub fn predict(&self, head: Series) -> Result<Prediction, MlError> {
        let end_t = head.last().ok_or(MlError::InsufficientData)?.0;
        let offset_t = -end_t;
        let shifted_head: Series = head.into_iter().map(|(t, x)| (t + offset_t, x)).collect();
        let align = self.align(&shifted_head, 0.0, 1.0, 60).ok_or(MlError::NoAlignment)?;
        let aligned_head: Series = shifted_head.into_iter().map(|(t, x)| (t + align.t, x + align.x)).collect();
        let t_head = aligned_head.last().ok_or(MlError::InsufficientData)?.0;
        let t_end = self.series.last().ok_or(MlError::InsufficientData)?.0;
        let x_end = self.series.last().ok_or(MlError::InsufficientData)?.1;
        Ok(Prediction {  t_head, head: aligned_head, path: self.clone(), t_end, x_end })
    }

    pub fn head(&self, to_t: f64) -> Option<Self> {
        let head: Series = self.series.iter().filter_map(|x| if x.0 < to_t { Some(*x) } else { None }).collect();
        Some(Self {
            offset_t: self.offset_t,
            offset_x: self.offset_x,
            spline: Self::create_monotonic_spline(&head)?,
            series: head,
        })
    }
        /// Second order.
    /// Assumes monotonicity of velocity function v(t) (and displacement function) (so once we start moving away from the target velocity we wont come back).
    fn search_v(series: SeriesRef, v_star: f64) -> Result<(f64, f64), MlError> {
        if series.len() < 3 {
            return Err(MlError::InsufficientData);
        }

        let mut dxdt_segs: Vec<(f64, f64)> = Vec::with_capacity(series.len() - 1);

        for i in 0..(series.len() - 1) {
            let (t0, x0) = series[i];
            let (t1, x1) = series[i + 1];
            dxdt_segs.push(((t1 + t0)/2.0 ,(x1 - x0)/(t1 - t0)));   
        }

        let mut target: Option<(usize, usize)> = None;
        for i in 0..(dxdt_segs.len() - 1) {
            if dxdt_segs[i].1 < v_star && dxdt_segs[i + 1].1 > v_star {
                target = Some((i, i+1));
                break;
            } else if dxdt_segs[i].1 > v_star && dxdt_segs[i + 1].1 < v_star {
                target = Some((i, i+1));
                break;
            } 
        }
        let (i0, i1) = target.ok_or(MlError::NoAlignment)?;

        let a = (dxdt_segs[i1].1 - dxdt_segs[i0].1)/(dxdt_segs[i1].0 - dxdt_segs[i0].0);

        let t_star = dxdt_segs[i0].0 + (v_star - dxdt_segs[i0].1) / a;

        let v_mid = dxdt_segs[i0].1 + a * (series[i1].0 - dxdt_segs[i0].0);

        let x_star = if t_star > series[i1].0 {
            let delta_t = t_star - series[i1].0;
            series[i1].1 + delta_t * v_mid + 0.5 * delta_t * delta_t * a
        } else {
            let delta_t = series[i1].0 - t_star;
            series[i1].1 - delta_t * v_mid - 0.5 * delta_t * delta_t * a
        };
        Ok((t_star, x_star))
    }

    fn search_spline(spline: &Spline<f64, f64>, v_star: f64, h: f64, delta: f64, n: usize) -> Result<(f64, f64), MlError> {
        let k = spline.keys()[spline.keys().len() / 2];
        let t = k.t;
        let s1 = spline.sample(t + h).ok_or(MlError::NoAlignment)?;
        let s0 = spline.sample(t).ok_or(MlError::NoAlignment)?;
        let vc = ((s1 - s0)/h - v_star).abs();
        Self::search_spline_n(spline, h, t, vc, v_star, delta, n)
    }

    fn search_spline_n(spline: &Spline<f64, f64>, h: f64, t: f64, vc: f64, v_star: f64, delta: f64, n: usize) -> Result<(f64, f64), MlError> {
        if n == 0 {
            return Ok((t, spline.sample(t).unwrap()));
        }

        let tup = t + delta;
        let tdown = t - delta;

        let su1 = spline.sample(tup + h);
        let su0 = spline.sample(tup);

        let vu = if let (Some(su1), Some(su0)) = (su1, su0) {
            Some(((su1 - su0)/h - v_star).abs())
        } else {
            None
        };

        let sd1 = spline.sample(tdown + h);
        let sd0 = spline.sample(tdown);

        let vd = if let (Some(sd1), Some(sd0)) = (sd1, sd0) {
            Some(((sd1 - sd0)/h - v_star).abs())
        } else {
            None
        };

        if let (Some(vu), Some(vd)) = (vu, vd) {
            if vu < vd && vu < vc {
                Self::search_spline_n(spline, h, tup, vu, v_star, delta, n-1)
            } else if vd < vu && vd < vc {
                Self::search_spline_n(spline, h, tdown, vd, v_star, delta, n-1)
            } else {
                Self::search_spline_n(spline, h, t, vc, v_star, delta/2.0, n-1)
            }
        } else {
            if let Some(vu) = vu {
                if vu < vc {
                    Self::search_spline_n(spline, h, tup, vu, v_star, delta, n-1)
                } else {
                    Self::search_spline_n(spline, h, t, vc, v_star, delta/2.0, n-1)
                }
            } else if let Some(vd) = vd {
                if vd < vc {
                    Self::search_spline_n(spline, h, tdown, vd, v_star, delta, n-1)
                } else {
                    Self::search_spline_n(spline, h, t, vc, v_star, delta/2.0, n-1)
                }
            } else {
                Self::search_spline_n(spline, h, t, vc, v_star, delta/2.0, n-1)
            }
        }
    }

    fn apply_offset(series: Series, offset_t: f64, offset_x: f64) -> Series {
        series.into_iter().map(|(t, x)| (t + offset_t, x + offset_x)).collect()
    }

    fn create_monotonic_spline(points: SeriesRef) -> Option<Spline<f64, f64>> {
        // Ensure there are at least two points
        if points.len() < 2 {
            return None;
        }
    
        let mut keys: Vec<Key<f64, f64>> = Vec::new();
    
        // Calculate slopes between consecutive points
        let n = points.len();
        let mut slopes = vec![0.0; n - 1];
    
        for i in 0..n - 1 {
            let dy = points[i + 1].1 - points[i].1;
            let dx = points[i + 1].0 - points[i].0;
            slopes[i] = dy / dx;
        }
    
        // Calculate tangents for each point
        let mut tangents = vec![0.0; n];

        tangents[0] = slopes[0];
        tangents[n - 1] = slopes[n - 2];
    
        for i in 1..n - 1 {
            tangents[i] = (slopes[i - 1] + slopes[i]) / 2.0;
    
            // Enforce monotonicity
            if tangents[i].signum() != slopes[i].signum() {
                tangents[i] = 0.0;
            } else {
                // Clamp the tangents to maintain monotonicity
                let a = tangents[i] / slopes[i];
                let b = tangents[i] / slopes[i - 1];
                if a > 3.0 || b > 3.0 {
                    tangents[i] = 3.0 * slopes[i].min(slopes[i - 1]);
                }
            }
        }
    
        // Generate keys for the spline
        for i in 0..n - 1 {
            let p0 = &points[i];
            let p1 = &points[i + 1];
            let t0 = p0.0;
            let t1 = p1.0;
            let h = t1 - t0;
            let v0 = p0.1;
            let v1 = p1.1;
    
            let m0 = tangents[i] * h;
            let m1 = tangents[i + 1] * h;
    
            keys.push(Key::new(t0, v0, splines::Interpolation::Linear));
        }
    
        Some(Spline::from_vec(keys))
    }

    pub fn align(&self, other: SeriesRef, align_start: f64, tick_t: f64, n: usize) -> Option<Alignment> {

        let err = self.align_err(other, align_start)?;
        let (offset_t, err) = self.align_n(other, align_start, err, tick_t, n)?;
        let offset_x = self.fit_x(other, offset_t)?;
        Some(Alignment { t: offset_t, x: offset_x, err })
    }  

    fn fit_x(&self, other: SeriesRef, offset_t: f64) -> Option<f64> {
        let mut vertical_sum = 0.0;
        let mut n = 0;

        for (t, x) in other.iter() {
            let new_t = *t + offset_t;
            if let Some(x_e) = self.sample(new_t) {
                vertical_sum += x_e - *x;
                n += 1;
            }
        }
        if n > 0 {
            Some(vertical_sum / n as f64)
        } else {
            None
        }
    }   
    
    fn align_n(&self, head: SeriesRef, align_t: f64, align_err: f64, tick_t: f64, n: usize) -> Option<(f64, f64)> {
        if n == 0 {
            return Some((align_t, align_err));
        }

        // Try aligning up
        let tu = align_t + tick_t;
        let td = align_t - tick_t;

        let au = self.align_err(head, tu);
        let ad = self.align_err(head, td);

        if let (Some(au), Some(ad)) = (au, ad) {
            if au < align_err && au < ad {
                self.align_n_up(head, tu, au, tick_t, n - 1)
            } else if ad < align_err && ad < au {
                self.align_n_down(head, td, ad, tick_t, n - 1)
            } else {
                self.align_n(head, align_t, align_err, tick_t / 2.0, n - 1)
            }
        } else {
            if let Some(au) = au {
                if au < align_err {
                    self.align_n_up(head, tu, au, tick_t, n - 1)
                } else {
                    self.align_n(head, align_t, align_err, tick_t / 2.0, n - 1)
                }
            } else if let Some(ad) = ad {
                if ad < align_err {
                    self.align_n_down(head, td, ad, tick_t, n - 1)
                } else {
                    self.align_n(head, align_t, align_err, tick_t / 2.0, n - 1)
                }
            } else {
                None
            }
            
        }
    }

    fn align_n_up(&self, head: SeriesRef, align_t: f64, align_err: f64, tick_t: f64, n: usize) -> Option<(f64, f64)> {
        if n == 0 {
            return Some((align_t, align_err));
        }

        // Try aligning up
        let tu = align_t + tick_t;

        if let Some(au) = self.align_err(head, tu) {

            if au < align_err {
                self.align_n_up(head, tu, au, tick_t, n - 1)
            } else {
                self.align_n(head, align_t, align_err, tick_t / 2.0, n - 1)
            }

        } else {
            self.align_n(head, align_t, align_err, tick_t / 2.0, n - 1)
        }
    }


    fn align_n_down(&self, head: SeriesRef, align_t: f64, align_err: f64, tick_t: f64, n: usize) -> Option<(f64, f64)> {
        if n == 0 {
            return Some((align_t, align_err));
        }
        // Try aligning up
        let td = align_t - tick_t;

        if let Some(ad) = self.align_err(head, td) {

            if ad < align_err {
                self.align_n_down(head, td, ad, tick_t, n - 1)
            } else {
                self.align_n(head, align_t, align_err, tick_t / 2.0, n - 1)
            }

        } else {
            self.align_n(head, align_t, align_err, tick_t / 2.0, n - 1)
        }
    }

    pub fn delta_t(&self) -> Option<f64> {
        Some(self.series.last()?.0 - self.series.first()?.0)
    }

    pub fn range(&self) -> Option<Range<f64>> {
        Some(self.series.first()?.0..self.series.last()?.0)
    }

    pub fn points(&self, range: Range<f64>) -> Series {
        self.series.iter().filter_map(|x| if range.contains(&x.0) { Some(*x) } else { None }).collect()
    }

    pub fn com_range(&self, range: Range<f64>) -> Option<f64> {
        let points = self.points(range);
        if points.is_empty() {
            None
        } else {
            Some(points.iter().map(|x| x.1).sum::<f64>() / points.len() as f64)
        }
    }

    pub fn sample(&self, t: f64) -> Option<f64> {
        self.spline.sample(t)
    }

    pub fn align_err(&self, series: SeriesRef, align_t: f64) -> Option<f64> {
        // Calculate vertial shift via centre of mass.
        let mut err = 0.0;
        let mut n = 0;
        for i in 0..series.len() - 1 {
            let delta_x = series[i + 1].1 - series[i].1;
            if let (Some(e0), Some(e1)) = (self.sample(series[i].0 + align_t), self.sample(series[i + 1].0 + align_t)) {
                let delta_e = e1 - e0;
                err += (delta_x - delta_e).powi(2);
                n += 1;
            }
        }
        if n > 0 {
            Some(err / n as f64)
        } else {
            None
        }
    }

    pub fn plot(&self, path: &str) {
        Self::plot_series(&[&self.series], path)

    }

    pub fn plot_series(series_list: &[SeriesRef], path: &str) {
        let mut max_t = None;
        let mut min_t = None;

        let mut max_x = None;
        let mut min_x = None;
        for series in series_list.iter() {
            // t is increasing.
            let tmax = series.last().unwrap().0.max(series.first().unwrap().0);
            let tmin = series.first().unwrap().0.min(series.last().unwrap().0);
            // since x is decreasing.
            let xmax = series.last().unwrap().1.max(series.first().unwrap().1);
            let xmin = series.last().unwrap().1.min(series.first().unwrap().1);
            if max_t.is_none() || Some(tmax) > max_t {
                max_t = Some(tmax);
            }
            if min_t.is_none() || Some(tmin) < min_t {
                min_t = Some(tmin);
            }
            if max_x.is_none() || Some(xmax) > max_x {
                max_x = Some(xmax);
            }
            if min_x.is_none() || Some(xmin) < min_x {
                min_x = Some(xmin);
            }
        }
        
        if let (Some(max_t), Some(min_t), Some(max_x), Some(min_x)) = (max_t, min_t, max_x, min_x) {
            use plotters::prelude::*;
            let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
            root.fill(&WHITE).unwrap();
    
            let mut chart = ChartBuilder::on(&root)
                .caption("ML Trainer", ("sans-serif", 50).into_font())
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(min_t - 1.0..max_t + 1.0, min_x - 1.0..max_x + 1.0).unwrap();
    
            chart.configure_mesh().draw().unwrap();
            
            for (i, series) in series_list.iter().enumerate() {
                let a = i + 24;
                let b = i * 4;
                let c = i * 74; 
                let col = RGBColor((a % 255) as u8, (b % 255) as u8, (c % 255) as u8);
                chart.draw_series(
                    series.iter().map(|(t, x)| Circle::new((*t, *x), 2, col))
                ).unwrap();
            }
            
    
            root.present().unwrap();
        }

    }
}

impl MlTrainer {
    pub fn new(v_star: f64) -> Self {
        Self {
            datasets: Vec::new(),
            v_star,
        }
    }

    /// Returns None if successful (as series is moved), else returns the series.
    /// Series must be oriented so that distance is increasing.
    pub fn add_series(&mut self, series: Series) -> Result<(), MlError> {
        self.datasets.push(Dataset::new_aligned(series, self.v_star)?);
        Ok(())
    }

    pub fn add_series_aligned(&mut self, series: Series, step: f64) -> Result<(), MlError> {
        if let Ok(d0) = self.generate_aggregate(step) {
            self.datasets.push(Dataset::new_aligned_to(series, &d0, self.v_star)?);
        } else {
            self.datasets.push(Dataset::new_aligned(series, self.v_star)?);
        }
        
        Ok(())
    }

    // pub fn predict(&mut self, head: Dataset) -> Option<Series> {
        
    // }
    
    pub fn reshape(self, v_star: f64) -> Result<Self, MlError> {
        let mut reshaped = Self::new(v_star);
        for set in self.datasets {
            reshaped.add_series(set.series)?;
        }
        Ok(reshaped)
    }

    pub fn reshape_aligned(self, v_star: f64, step: f64) -> Result<Self, MlError> {
        let mut reshaped = Self::new(v_star);
        for set in self.datasets {
            let _ = reshaped.add_series_aligned(set.series, step);
        }
        Ok(reshaped)
    }

    pub fn predict_aggregate(&self, t: f64) -> Option<f64> {
        let mut n = 0;
        let mut sum = 0.0;
        for set in self.datasets.iter() {
            if let Some(v) = set.sample(t) {
                sum += v;
                n += 1;
            }
        }
        if n > 0 {
            Some(sum / n as f64) 
        } else {
            None
        }
    }

    pub fn generate_aggregate(&self, step: f64) -> Result<Dataset, MlError> {
        let range = self.range_t().ok_or(MlError::InsufficientData)?;
        let mut t = range.start;
        let mut agg_series = Series::new();
        while t < range.end {
            if let Some(x) = self.predict_aggregate(t) {
                agg_series.push((t, x))
            } 
            t += step;
        }

        Dataset::new_aligned(agg_series, self.v_star)
    }

    pub fn range_t(&self) -> Option<Range<f64>> {
        let mut max_t = None;
        let mut min_t = None;

        for set in self.datasets.iter() {
            // t is increasing.
            let tmax = set.series.last().unwrap().0;
            let tmin = set.series.first().unwrap().0;

            if max_t.is_none() || Some(tmax) > max_t {
                max_t = Some(tmax);
            }
            if min_t.is_none() || Some(tmin) < min_t {
                min_t = Some(tmin);
            }
        }

        if let (Some(max_t), Some(min_t)) = (max_t, min_t) {
            Some((min_t..max_t))
        } else {
            None
        }
    }

    pub fn plot(&self, path: &str) {
        let mut max_t = None;
        let mut min_t = None;

        let mut max_x = None;
        let mut min_x = None;
        for set in self.datasets.iter() {
            // t is increasing.
            let tmax = set.series.last().unwrap().0;
            let tmin = set.series.first().unwrap().0;
            // since x is decreasing.
            let xmax = set.series.first().unwrap().1;
            let xmin = set.series.last().unwrap().1;
            if max_t.is_none() || Some(tmax) > max_t {
                max_t = Some(tmax);
            }
            if min_t.is_none() || Some(tmin) < min_t {
                min_t = Some(tmin);
            }
            if max_x.is_none() || Some(xmax) > max_x {
                max_x = Some(xmax);
            }
            if min_x.is_none() || Some(xmin) < min_x {
                min_x = Some(xmin);
            }
        }
        
        if let (Some(max_t), Some(min_t), Some(max_x), Some(min_x)) = (max_t, min_t, max_x, min_x) {
            use plotters::prelude::*;
            let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
            root.fill(&WHITE).unwrap();
    
            let mut chart = ChartBuilder::on(&root)
                .caption("ML Trainer", ("sans-serif", 50).into_font())
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(min_t - 1.0..max_t + 1.0, min_x - 1.0..max_x + 1.0).unwrap();
    
            chart.configure_mesh().draw().unwrap();
            
            for (i, set) in self.datasets.iter().enumerate() {
                let a = i + 24;
                let b = i * 4;
                let c = i * 74; 
                let col = RGBColor((a % 255) as u8, (b % 255) as u8, (c % 255) as u8);
                chart.draw_series(
                    set.series.iter().map(|(t, x)| Circle::new((*t, *x), 2, col))
                ).unwrap();
            }
            
    
            root.present().unwrap();
        }

    }


}


#[test]
fn test_search() {
    let series = vec![
        (0.0, 7.0),
        (1.0, 5.0),
        (2.0, 2.0),
    ];
    let v_star = Dataset::search_v(&series, -2.3);

    println!("{:?}", v_star.unwrap());
}