use core::num;
use std::time::SystemTime;

use nalgebra::SVector;
use num_traits::Pow;


#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DynamicState {
    pub t: f64,
    pub dis: f64,
    pub vel: f64,
    pub acc: f64,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PlateState {
    pub t: f64,
    pub dis: f64,
    pub vel: f64,
    pub acc: f64,
}

impl PlateState {
    pub fn new(t_0: f64, dis: f64, vel: f64, acc: f64) -> Self {
        Self {
            dis, vel, acc, t: t_0,
        }
    }

    pub fn predict(self, dt: f64) -> Self {
        if self.vel == 0.0 {
            return self;
        }
        let mut vel = self.vel + self.acc * dt * self.vel.signum();
        let mut dis = self.dis + self.vel * dt + 0.5 * self.acc * dt * dt * self.vel.signum();
        if vel.is_sign_positive() != self.vel.is_sign_positive() {
            vel = 0.0;
            dis = self.dis;
        }
        Self {
            dis, vel, acc: self.acc, t: self.t + dt,
        }
    }

    pub fn approximate(self, delta: f64, approx_step: f64) -> Self {
        if self.vel == 0.0 {
            return self;
        }
        
        self.predict(delta)
    }

    pub fn velocity_for(&self, dis: f64, t: f64) -> f64 {
        let ddis = dis - self.dis;
        let delta = t - self.t;
        let u = (ddis - (0.5 * self.acc * delta * delta * self.vel.signum())) / delta;

        let v = u + self.acc * delta * self.vel.signum();

        v
    }

    // discount factor must be between 0 and 1.
    pub fn update(self, real_pos: f64, delta: f64, approx_step: f64, discount_factor: f64) -> Self {
        let predicted = self.approximate(delta, approx_step);

        // Set pos to the weighted mean of predicted and real.
        let real_dis = closest_to(predicted.dis, real_pos);

        let new_t = self.t + delta;

        let estimated_vel = self.velocity_for(real_dis, new_t);
        
        let vel_diff = estimated_vel - predicted.vel;

        // the longer the distance, the less we want this to affect the approx speed. 
        // Basically scale vel affect by the proportion of this delta to the total sim time (<= 1).
        let new_vel = predicted.vel + vel_diff * discount_factor;

        let dfactor = discount_factor.sqrt();
        // Weight new and old depending on contribution of current update to the whole state.
        let new_dis = predicted.dis * (1.0 - dfactor) + real_dis * dfactor;

        Self { dis: new_dis, vel: new_vel, acc: predicted.acc, t: new_t }
    }

    pub fn local_pos(&self, dis: f64) -> f64 {
        dis_to_pos(dis - self.dis)
    }

    pub fn slot_at_local_pos(pos: f64, n: usize) -> usize {
        let ds = (std::f64::consts::PI * 2.0) / n as f64;
        (pos / ds) as usize
    }

    pub fn local_pos_at_slot(slot: usize, n: usize) -> f64 {
        slot as f64 * (std::f64::consts::PI * 2.0) / n as f64
    }
}

impl DynamicState {
    pub fn new(t_0: f64, dis: f64, vel: f64, acc: f64) -> Self {
        Self {
            dis,
            vel,
            acc,
            t: t_0,
        }
    }

    pub fn from_delta(t_0: f64, dis: f64, delta: f64, approx_step: f64, acc: f64) -> DynamicState {
        DynamicState { dis, vel: dis / delta, acc, t: t_0 }
    }

    pub fn with_acc(self, acc: f64) -> Self {
        Self {
            acc,
            dis: self.dis,
            t: self.t, 
            vel: self.vel,
        }
    }

    /// ******************* The objective function ********************
    pub fn predict(self, dt: f64, a: f64, k: f64, att: f64, w: SVector<f64, 8>) -> DynamicState {
        if self.vel == 0.0 {
            return self;
        }
        let vabs = self.vel.abs();
        let acc = a + k * vabs.powf(att) + vabs * w[0] + vabs.pow(2) * w[1] + vabs.pow(3) * w[2] + vabs.pow(1.5) * w[3] + vabs.abs().powf(1.75) * w[4] + vabs.abs().powf(1.125) * w[5] + vabs.powf(2.5) * w[6] + vabs.powf(3.5) * w[7]; // Approximately proportional to velocity, we don't need to be super accurate.
        let mut vel: f64 = self.vel + acc * dt * self.vel.signum(); // Approximating a as a constant should be close enough given small enough dt.
        let mut dis = self.dis + self.vel * dt + 0.5 * acc * dt * dt * self.vel.signum();
        if vel.is_sign_positive() != self.vel.is_sign_positive() {
            vel = 0.0;
            dis = self.dis;
        }
        DynamicState { dis, vel, acc, t: self.t + dt }
    }

    pub fn approximate(self, delta: f64, approx_step: f64, a: f64, k: f64, att: f64, w: SVector<f64, 8>) -> DynamicState {
        if self.vel == 0.0 {
            return self;
        }
        let n = ((delta / approx_step) as usize).max(1);
        let dt = delta / n as f64;

        (0..n).into_iter().fold(self, |v, _| v.predict(dt, a, k, att, w))
    }

    pub fn update(self, real_pos: f64, delta: f64, approx_step: f64, a: f64, k: f64, att: f64, w: SVector<f64, 8>, discount_factor: f64) -> DynamicState {
        let predicted = self.approximate(delta, approx_step, a, k, att, w);

        // Set pos to the weighted mean of predicted and real.
        let real_dis = closest_to(predicted.dis, real_pos);

        let vel_modif = ((real_dis - predicted.dis) / delta) * self.vel.abs().sqrt();

        // the longer the distance, the less we want this to affect the approx speed. 
        // Basically scale vel affect by the proportion of this delta to the total sim time (<= 1).
        let new_vel = predicted.vel + vel_modif * discount_factor * 0.2;

        // Weight new and old depending on contribution of current update to the whole state.
        let dfactor = discount_factor.sqrt();
        let new_dis = predicted.dis * (1.0 - dfactor) + real_dis * dfactor;

        Self { dis: new_dis, vel: new_vel, acc: predicted.acc, t: self.t + delta }
    }

    // Return time, pos.
    pub fn solve(self, min_vel: f64, step: f64, a: f64, k: f64, att: f64, w: SVector<f64, 8>) -> Option<Self> {
        let mut t = 0.0;
        let mut state_buf = self.predict(step, a, k, att, w);
        for i in 0..1000000 {
            if state_buf.vel.abs() < min_vel.abs() {
                return Some(state_buf);
            }
            state_buf = state_buf.predict(step, a, k, att, w);
            t += step;
        }
        None
    }
}

fn closest_to(dis: f64, pos: f64) -> f64 {
    let dpos = dis_to_pos(dis);
    let delta = pos - dpos;

    if delta >= -std::f64::consts::PI && delta <= std::f64::consts::PI {
        dis + delta
    } else if delta > std::f64::consts::PI {
        dis - (std::f64::consts::PI * 2.0 - delta)
    } else {
        dis + (std::f64::consts::PI * 2.0 + delta)
    }
}

pub fn dis_to_pos(pos: f64) -> f64 {
    let n = pos % (std::f64::consts::PI * 2.0);
    if pos < 0.0 {
        (std::f64::consts::PI * 2.0) + n
    } else {
        n
    }
    
}