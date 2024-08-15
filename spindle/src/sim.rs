use std::time::{SystemTime, Duration};

use nalgebra::{ComplexField, SVector, VectorN};

use crate::game::*;

pub struct Prediction {
    pub point: f64,
}

#[derive(Debug, Clone)]
pub struct SimState {
    pub plate_state: Option<PlateState>,
    pub dynamic_state: Option<DynamicState>,
    pub clockwise: bool,
    pub plate_clicks: Vec<(SystemTime, f64, Option<PlateState>)>,
    pub dynamic_clicks: Vec<(SystemTime, f64, Option<DynamicState>)>,
    pub start: SystemTime,
    pub step: f64,
    pub params: SimParams,
}

#[derive(Debug, Copy, Clone)]
pub struct SimParams {
    pub plate_acc: f64,
    pub dynamic_acc: f64,
    pub min_vel: f64,
    pub k: f64,
    pub att: f64,
    pub dynamic_weights: SVector<f64, 8>,
    pub end_t: f64,
    pub end_d: f64,
}

impl SimParams {
    pub fn train_on(self, other: &SimParams, weight: f64) -> Self {
        assert!(weight <= 1.0 && weight >= 0.0);
        let w1 = 1.0 - weight;
        let w2 = weight;
        Self {
            plate_acc: self.plate_acc * w1 + other.plate_acc * w2,
            dynamic_acc: self.dynamic_acc * w1 + other.dynamic_acc * w2,
            min_vel: self.min_vel * w1 + other.min_vel * w2,
            k: self.k * w1 + other.k * w2,
            att: self.att * w1 + other.att * w2,
            end_t: self.end_t * w1 + other.end_t * w2,
            end_d: self.end_d * w1 + other.end_d * w2,
            dynamic_weights: w1 * self.dynamic_weights + w2 * other.dynamic_weights,
        }
    }

}

impl SimState {
    pub fn new(clockwise: bool, start: SystemTime, step: f64, params: SimParams) -> Self {
        Self {
            plate_state: None,
            dynamic_state: None,
            clockwise,
            plate_clicks: Vec::new(),
            dynamic_clicks: Vec::new(),
            start,
            step,
            params,
        }
    }

    pub fn dynamic_states(&self) -> Vec<DynamicState> {
        self.dynamic_clicks.iter().filter_map(|(_, _, ds)|*ds).collect()
    }

    pub fn plate_states(&self) -> Vec<PlateState> {
        self.plate_clicks.iter().filter_map(|(_, _, ps)|*ps).collect()
    }

    pub fn directional_plate(&self) -> f64 {
        if self.clockwise {
            1.0
        } else {
            -1.0
        }
    }

    pub fn directional_dynamic(&self) -> f64 {
        if self.clockwise {
            -1.0
        } else {
            1.0
        }
    }

    pub fn directional_end_d(&self) -> f64 {
        if self.clockwise {
            -self.params.end_d
        } else {
            self.params.end_d
        }
    }

    fn time(&self, st: SystemTime) -> f64 {
        st.duration_since(self.start).unwrap().as_secs_f64()
    }

    pub fn plate_dis(&self, pos_0: f64, pos_1: f64) -> f64 {
        if pos_0 == pos_1 {
            return std::f64::consts::PI * 2.0;
        }
        if self.clockwise {
            if pos_1 < pos_0 {
                pos_1 + std::f64::consts::PI * 2.0
            } else {
                pos_1 - pos_0
            }
        } else {
            if pos_1 > pos_0 {
                pos_0 + (std::f64::consts::PI * 2.0 - pos_1)
            } else {
                pos_0 - pos_1
            }
        }
    }

    pub fn dynamic_dis(&self, pos_0: f64, pos_1: f64) -> f64 {
        if pos_0 == pos_1 {
            return std::f64::consts::PI * 2.0;
        }
        if !self.clockwise {
            if pos_1 < pos_0 {
                pos_1 + std::f64::consts::PI * 2.0
            } else {
                pos_1 - pos_0
            }
        } else {
            if pos_1 > pos_0 {
                pos_0 + (std::f64::consts::PI * 2.0 - pos_1)
            } else {
                pos_0 - pos_1
            }
        }
    }

    pub fn plate_click(&mut self, time: SystemTime, click_pos: f64, min_discount: f64) {
        if let Some((last, lpos, lps)) = self.plate_clicks.last() {
            let delta = time.duration_since(*last).unwrap().as_secs_f64();
            match self.plate_state {
                Some(ps) => {
                    let d = time.duration_since(self.system_time(ps.t)).unwrap().as_secs_f64();
                    let discount_factor = f64::max(1.0 / self.plate_clicks.len() as f64, min_discount);
                    self.plate_state = Some(ps.update(click_pos, d, self.step, discount_factor))
                },
                None => {
                    let av = self.plate_dis(*lpos, click_pos) / delta;
                    let v_end: f64 = av + self.params.plate_acc * delta * 0.5;
                    println!("av: {}, v_end: {}", av, v_end);
                    self.plate_state = Some(PlateState::new(self.time(time), click_pos, v_end * self.directional_plate(), self.params.plate_acc))
                },
            }
            self.plate_clicks.push((time, click_pos, self.plate_state));
        } else {
            self.plate_clicks.push((time, click_pos, None));
        }
    }

    // pub fn dynamic_click(&mut self, time: SystemTime, click_pos: f64, min_discount: f64) {
    //     if let Some((last, lpos, lds)) = self.dynamic_clicks.last() {
    //         let delta = time.duration_since(*last).unwrap().as_secs_f64();
    //         match self.dynamic_state {
    //             Some(ds) => {
    //                 let d = time.duration_since(self.system_time(ds.t)).unwrap().as_secs_f64();
    //                 let discount_factor = f64::max(1.0 / self.dynamic_clicks.len() as f64, min_discount);
    //                 self.dynamic_state = Some(ds.update(click_pos, d, self.step, self.params.dynamic_acc, self.params.k, self.params.att, self.params.dynamic_weights, discount_factor))
    //             },
    //             None => {
    //                 let av = self.dynamic_dis(*lpos, click_pos) / delta;
    //                 let acc: f64 = self.params.dynamic_acc + self.params.k * av;
    //                 let v_end: f64 = av + acc * delta * 0.5;
    //                 self.dynamic_state = Some(DynamicState::new(self.time(time), click_pos, v_end * self.directional_dynamic(), self.params.dynamic_acc))
    //             },
    //         }
    //         self.dynamic_clicks.push((time, click_pos, self.dynamic_state));
    //     } else {
    //         self.dynamic_clicks.push((time, click_pos, None));
    //     }
    // }

    pub fn dynamic_click(&mut self, time: SystemTime, click_pos: f64, min_discount: f64) {
        if let Some((last, lpos, lds)) = self.dynamic_clicks.last() {
            let delta = time.duration_since(*last).unwrap().as_secs_f64();
            match self.dynamic_state {
                Some(ds) => {
                    let d = time.duration_since(self.system_time(ds.t)).unwrap().as_secs_f64();
                    let discount_factor = f64::max(1.0 / self.dynamic_clicks.len() as f64, min_discount);
                    self.dynamic_state = Some(ds.update(click_pos, d, self.step, self.params.dynamic_acc, self.params.k, self.params.att, self.params.dynamic_weights, discount_factor))
                },
                None => {
                    let d0 = *lpos;
                    let d1 = *lpos + self.dynamic_dis(*lpos, click_pos) * self.directional_dynamic();
                    let mut initial_state = self.initial_dynamic_state(delta, d0, d1, 40);
                    initial_state.t = self.time(time);
                    self.dynamic_state = Some(initial_state)
                },
            }
            self.dynamic_clicks.push((time, click_pos, self.dynamic_state));
        } else {
            self.dynamic_clicks.push((time, click_pos, None));
        }
    }


    /// Calculates the veolcity of the dynamic from the time interval to travel d distance. The d values must be overall angular displacement distatance not a normalised angle.
    pub fn initial_dynamic_state(&self, delta: f64, d0: f64, d1: f64, n: usize) -> DynamicState {
        let dir = (d1 - d0).signum();
        let cvel = d1 - d0 / delta;
        let ds = DynamicState::new(0.0, d0, cvel, self.params.dynamic_acc * dir);
        let out = ds.approximate(delta, self.step, self.params.dynamic_acc, self.params.k, self.params.att, self.params.dynamic_weights);
        let cerr = (d1 - out.dis).abs();

        self.dyvel_rec(delta, d0, d1, out, cvel, cerr, 1.0, n)
    }

    fn dyvel_rec(&self, delta: f64, d0: f64, d1: f64, cstate: DynamicState, cvel: f64, cerr: f64, delta_v: f64, n: usize) -> DynamicState {
        if n == 0 {
            return cstate;
        }

        let dir = (d1 - d0).signum();

        let upvel = cvel + delta_v;
        let dup = DynamicState::new(0.0, d0, upvel, self.params.dynamic_acc * dir);

        let downvel = cvel - delta_v;
        let ddown = DynamicState::new(0.0, d0, downvel, self.params.dynamic_acc * dir);

        let outup = dup.approximate(delta, self.step, self.params.dynamic_acc, self.params.k, self.params.att, self.params.dynamic_weights);
        let outuperr = (d1 - outup.dis).abs();
        
        let outdown = ddown.approximate(delta, self.step, self.params.dynamic_acc, self.params.k, self.params.att, self.params.dynamic_weights);
        let outdownerr = (d1 - outdown.dis).abs();
        
        if outuperr < cerr && outuperr < outdownerr {
            self.dyvel_rec(delta, d0, d1, outup, upvel, outuperr, delta_v, n - 1)
        } else if outdownerr < cerr && outdownerr < outuperr {
            self.dyvel_rec(delta, d0, d1, outdown, downvel, outdownerr, delta_v, n - 1)
        } else {
            self.dyvel_rec(delta, d0, d1, cstate, cvel, cerr, delta_v / 2.0, n - 1)
        }
    }
        

    pub fn solve(&self) -> Option<f64> {
        // Find when velocity drops below min.
        let ds = self.dynamic_state?.solve(self.params.min_vel, self.step, self.params.dynamic_acc, self.params.k, self.params.att, self.params.dynamic_weights)?;
        self.finalize(ds)
    }

    pub fn finalize(&self, ds: DynamicState) -> Option<f64> {
        let final_pos = dis_to_pos(ds.dis + self.directional_end_d());
        let final_t = ds.t + self.params.end_t;
        let delta = final_t - self.plate_state?.t;
        if delta > 0.0 {
            let ps = self.plate_state?.approximate(delta, self.step); 
            Some(ps.local_pos(final_pos))
        } else {
            None
        }
    }

    pub fn system_time(&self, t: f64) -> SystemTime {
        self.start + Duration::from_secs_f64(t)
    }

    pub fn predict_plate(&self, time: SystemTime) -> Option<PlateState> {
        if let Some(ps) = &self.plate_state {
            let delta = time.duration_since(self.system_time(ps.t)).unwrap().as_secs_f64();
            Some(ps.approximate(delta, self.step))
        } else {
            None
        }
    }

    pub fn predict_dynamic(&self, time: SystemTime) -> Option<DynamicState> {
        if let Some(ds) = self.dynamic_state {
            if ds.vel == 0.0 {
                return Some(ds);
            }
            let delta = time.duration_since(self.system_time(ds.t)).unwrap().as_secs_f64();
            Some(ds.approximate(delta, self.step, self.params.dynamic_acc, self.params.k, self.params.att, self.params.dynamic_weights))
        } else {
            None
        }
    }

    pub fn ssr_dynamic(&self, params: &SimParams, init_state: &DynamicState) -> (usize, f64, f64) {
        let mut ssr_t = 0.0;
        let mut ssr_d = 0.0;
        let mut n: usize = 0;
        for (_, _, state) in self.dynamic_clicks.iter() {
            if let Some(state) = state {
                if state.t > init_state.t && state.dis.abs() > init_state.dis.abs() {
                    let delta = state.t - init_state.t;
                    let pr_state = init_state.approximate(delta, self.step, params.dynamic_acc, params.k, params.att, params.dynamic_weights);

                    let e_t = state.t - pr_state.t;
                    let e_t_sq = e_t * e_t;
                    ssr_t += e_t_sq;

                    let e_d = state.dis- pr_state.dis;
                    let e_d_sq = e_d * e_d;
                    ssr_d += e_d_sq;

                    n += 1;

                }
            }
        }
        (n, ssr_t, ssr_d)
    }

    pub fn train_a(&self, params: SimParams, init_state: &DynamicState, trials: usize, delta_a: f64, delta_k: f64, delta_att: f64, delta_w: [f64; 8]) -> SimParams {
        if trials == 0 {
            return params;
        }

        let (_, _, ssr_d) = self.ssr_dynamic(&params, init_state);

        let mut up_params = params;
        up_params.dynamic_acc += delta_a;

        let (_, _, ssr_d_up) = self.ssr_dynamic(&up_params, init_state);

        let mut down_params = params;
        down_params.dynamic_acc -= delta_a;

        let (_, _, ssr_d_down) = self.ssr_dynamic(&down_params, init_state);

        if ssr_d_up < ssr_d && ssr_d_up < ssr_d_down {
            self.train_w(up_params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w, 0)
        } else if ssr_d_down < ssr_d && ssr_d_down < ssr_d_up {
            self.train_w(down_params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w, 0)
        } else {
            self.train_w(params, init_state, trials - 1, delta_a / 2.0, delta_k, delta_att, delta_w, 0)
        }
    }

    pub fn train_w(&self, params: SimParams, init_state: &DynamicState, trials: usize, delta_a: f64, delta_k: f64, delta_att: f64, mut delta_w: [f64; 8], wn: usize) -> SimParams {
        if trials == 0 {
            return params;
        }
        if wn >= 8 {
            return self.train_k(params, init_state, trials, delta_a, delta_k, delta_att, delta_w);
        }

        let (_, _, ssr_d) = self.ssr_dynamic(&params, init_state);

        let mut up_params = params;
        up_params.dynamic_weights[wn] += delta_w[wn];

        let (_, _, ssr_d_up) = self.ssr_dynamic(&up_params, init_state);

        let mut down_params = params;
        down_params.dynamic_weights[wn] -= delta_w[wn];

        let (_, _, ssr_d_down) = self.ssr_dynamic(&down_params, init_state);

        if ssr_d_up < ssr_d && ssr_d_up < ssr_d_down {
            self.train_w(up_params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w, wn + 1)
        } else if ssr_d_down < ssr_d && ssr_d_down < ssr_d_up {
            self.train_w(down_params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w, wn + 1)
        } else {
            delta_w[wn] /= 2.0;
            self.train_w(params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w, wn + 1)
        }
    }


    pub fn train_k(&self, params: SimParams, init_state: &DynamicState, trials: usize, delta_a: f64, delta_k: f64, delta_att: f64, delta_w: [f64; 8]) -> SimParams {
        if trials == 0 {
            return params;
        }

        let (_, _, ssr_d) = self.ssr_dynamic(&params, init_state);

        let mut up_params = params;
        up_params.k += delta_k;

        let (_, _, ssr_d_up) = self.ssr_dynamic(&up_params, init_state);

        let mut down_params = params;
        down_params.k -= delta_k;

        let (_, _, ssr_d_down) = self.ssr_dynamic(&down_params, init_state);

        if ssr_d_up < ssr_d && ssr_d_up < ssr_d_down {
            self.train_att(up_params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w)
        } else if ssr_d_down < ssr_d && ssr_d_down < ssr_d_up {
            
            self.train_att(down_params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w)
        } else {
            self.train_att(params, init_state, trials - 1, delta_a, delta_k / 2.0, delta_att, delta_w)
        }
    }

    pub fn train_att(&self, params: SimParams, init_state: &DynamicState, trials: usize, delta_a: f64, delta_k: f64, delta_att: f64, delta_w: [f64; 8]) -> SimParams {
        if trials == 0 {
            return params;
        }

        let (_, _, ssr_d) = self.ssr_dynamic(&params, init_state);

        let mut up_params = params;
        up_params.att += delta_att;

        let (_, _, ssr_d_up) = self.ssr_dynamic(&up_params, init_state);

        let mut down_params = params;
        down_params.att -= delta_att;

        let (_, _, ssr_d_down) = self.ssr_dynamic(&down_params, init_state);

        if ssr_d_up < ssr_d && ssr_d_up < ssr_d_down {
            self.train_a(up_params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w)
        } else if ssr_d_down < ssr_d && ssr_d_down < ssr_d_up {
            self.train_a(down_params, init_state, trials - 1, delta_a, delta_k, delta_att, delta_w)
        } else {
            self.train_a(params, init_state, trials - 1, delta_a, delta_k, delta_att / 2.0, delta_w)
        }
    }

    pub fn train(&self, weight: f64, trials: usize, delta_a: f64, delta_k: f64, delta_att: f64) -> Option<SimParams> {
        let outcome = self.train_a(self.params, self.dynamic_states().first()?, trials, delta_a, delta_k, delta_att, [0.05; 8]);
        Some(self.params.train_on(&outcome, weight))
    }


    pub fn plot(&self, params: &SimParams, path: &str) {
        let mut base: Option<DynamicState> = None;
        let mut n: usize = 0;
        let mut prs = Vec::new();
        let mut acts = Vec::new();
        let mut max_y: f64 = 0.0;
        for (_, _, state) in self.dynamic_clicks.iter() {
            if let Some(state) = state {
                if let Some(base) = base {
                    let delta = state.t - base.t;
                    let pr_state = base.approximate(delta, self.step, params.dynamic_acc, params.k, params.att, params.dynamic_weights);

                    acts.push((delta, state.dis.abs()));
                    prs.push((delta, pr_state.dis.abs()));

                    max_y = max_y.max(state.dis.abs().max(pr_state.dis.abs()));

                    n += 1;
                } else {
                    base = Some(*state);
                }
            }
        }

        use plotters::prelude::*;
        let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption("Scatter Plot Example", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..10.0, 0.0..max_y + 5.0).unwrap();

        chart.configure_mesh().draw().unwrap();

        chart.draw_series(
            prs.iter().map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())),
        ).unwrap();


        chart.draw_series(
            acts.iter().map(|(x, y)| Circle::new((*x, *y), 5, RED.filled())),
        ).unwrap();
        root.present().unwrap();

    }
    // pub fn estimate_params(&self) -> SimParams {
        
    // }
}

pub fn begin_sim_unsafe(callback: unsafe extern "C" fn(SimState)) {

}