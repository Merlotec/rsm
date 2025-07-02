use autopilot::{bitmap, geometry::Point};
use bevy::input::mouse::MouseButtonInput;
use bevy::window::WindowResolution;
use bevy::{
    input::{keyboard::KeyboardInput, ButtonState},
    prelude::*,
};
use bevy_debug_text_overlay::{screen_print, OverlayPlugin};
use enigo::Mouse;
use image::GenericImageView;
use motion::SmartTrigger;
use nalgebra::SVector;
use numread::NumberViewer;
use rodio::{source::SineWave, OutputStream, Source};
use spindle::game::dis_to_pos;
use spindle::ml::{Dataset, MlTrainer, MlWeights, Prediction, Series};
use spindle::sim::{Outcome, Solution, LAYOUT};
use spindle::{
    game::{DynamicState, PlateState},
    sim::{SimParams, SimState},
};
use std::sync::Arc;
use std::{
    ops::Deref,
    process::exit,
    sync::{
        mpsc::{Receiver, Sender},
        Mutex,
    },
    time::{Duration, SystemTime},
};

//pub mod auto;
pub mod motion;
pub mod numread;
//pub mod video;

#[derive(Resource, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    dynamic_acc: f64,
    plate_acc: f64,
    k: f64,
    att: f64,
    dynamic_weights: [f64; 8],
    min_vel: f64,
    end_t: f64,
    end_d: f64,
    dynamic_rect: [i32; 4],
    dynamic_rect_pos: f64,
    dynamic_points: Vec<f64>,
    plate_points: Vec<f64>,
    dynamic_color: [f32; 3],
    dynamic_threshold: f32,
    plate_color: [f32; 3],
    plate_threshold: f32,
    dynamic_pattern: Vec<[f32; 2]>,
    plate_pattern: Vec<[f32; 2]>,
    plate_rects: Vec<([i32; 4], f64)>,
    board_rad: f64,
    autopilot_rect: [i32; 4],
    autopilot_stby_col: [u8; 3],
    autopilot_trigger_col: [u8; 3],
    autopilot_sensitivity: f64,
    solve_vel: f64,
    bet_vel: f64,
    result_rect: [i32; 4],
    heartbeat_coords: [i32; 2],
    undo_coords: [i32; 2],
    weights: Vec<MlWeights>,
}

impl Settings {
    pub fn write_params(&mut self, path: &str, params: SimParams, final_vel: Option<f64>) {
        self.dynamic_acc = params.dynamic_acc;
        self.plate_acc = params.plate_acc;
        self.att = params.att;
        self.min_vel = params.min_vel;
        self.end_t = params.end_t;
        self.end_d = params.end_d;
        self.k = params.k;
        self.dynamic_weights = params.dynamic_weights.into();
        if let Some(final_vel) = final_vel {
            self.min_vel = final_vel;
        }

        self.write(path);
    }

    pub fn write(&self, path: &str) {
        if let Ok(str) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, str);
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Bet(Vec<u8>);

impl Bet {
    pub fn surrounding(i: u8, count: usize) -> Bet {
        let i = i as i16;
        let mut buf = vec![LAYOUT[i as usize]];
        let mut dis: i16 = 1;
        for _ in 0..(count - 1) {
            if dis > 0 {
                buf.push(LAYOUT[((i + dis as i16) % LAYOUT.len() as i16) as usize]);
                dis = -dis;
            } else {
                let j = {
                    if i + dis < 0 {
                        LAYOUT.len() + (i + dis) as usize
                    } else {
                        (i + dis) as usize % LAYOUT.len()
                    }
                };
                buf.push(LAYOUT[j]);
                dis = -dis;
                dis += 1;
            }
        }

        Self(buf)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimRecord {
    sln: Solution,
    outcome: Outcome,
    bet: Option<Bet>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Resource)]
pub struct HistoryRes(Vec<SimRecord>);

fn default_settings() -> Settings {
    Settings {
        plate_acc: -0.00321,
        dynamic_acc: -0.006,
        min_vel: 2.865527103538936,
        k: -0.0073,
        att: 2.5,
        end_t: 0.6,
        end_d: 1.3,
        dynamic_points: vec![
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2 * 3.0,
        ],
        plate_points: vec![
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2 * 3.0,
        ],
        dynamic_color: [0.9, 0.9, 0.9],
        dynamic_threshold: 0.2,
        plate_color: [0.2, 0.45, 0.2],
        plate_threshold: 0.2,
        dynamic_pattern: vec![[0.0, -270.0], [270.0, 0.0], [0.0, 270.0], [-270.0, 0.0]],
        plate_pattern: vec![[0.0, -150.0], [150.0, 0.0], [0.0, 150.0], [-150.0, 0.0]],
        dynamic_weights: [
            -0.14191429165750552,
            -0.022630863258615193,
            0.8132336094975379,
            -0.010149678768357285,
            0.41897401194437256,
            0.013894009590149026,
            0.0007739543914794932,
            0.0001279830932617193,
        ],
        dynamic_rect: [903, 235, 120, 50],
        dynamic_rect_pos: -389.0,
        plate_rects: vec![
            ([1031, 442, 80, 80], 0.0),
            ([1480, 480, 80, 80], std::f64::consts::FRAC_PI_2),
        ],
        board_rad: 657.0,
        autopilot_rect: [1386, 167, 13, 10],
        autopilot_stby_col: [201, 181, 172],
        autopilot_trigger_col: [48, 32, 1],
        autopilot_sensitivity: 0.2,
        solve_vel: 20.0,
        bet_vel: 14.0,
        result_rect: [836, 817, 31, 28],
        heartbeat_coords: [1100, 960],
        undo_coords: [1220, 1015],
        weights: Vec::new(),
    }
}

fn load_settings() -> Settings {
    let settings_f = std::fs::read_to_string("settings.json");

    let mut f = true;

    let settings: Settings = {
        if let Ok(s) = settings_f {
            if let Ok(j) = serde_json::from_str(&s) {
                f = false;
                j
            } else {
                println!("Failed to deserialize settings");
                default_settings()
            }
        } else {
            println!("No settings file found, creating one");
            default_settings()
        }
    };

    if f {
        if let Ok(str) = serde_json::to_string(&settings) {
            let _ = std::fs::write("settings.json", str);
        }
    }
    settings
}

fn main() {
    let trainer = if let Ok(s) = std::fs::read_to_string("trainer.json") {
        serde_json::from_str(&s).unwrap_or_default()
    } else {
        println!("(Re)creating trainer!");
        MlTrainer::default()
    };

    // trainer.plot("ml_pre.png");

    // let agg = trainer.generate_aggregate(0.5).unwrap();
    // agg.plot(true, "ml_agg.png");

    // let ext = agg.extrapolate_higher_deriv(vec![0.1, 0.1], 12, 0.5, 11.0).unwrap();
    // ext.plot(true, "ml_ext.png");

    // trainer.plot("ml.png");

    // let mut reshaped = trainer.reshape_aligned(-6.0, 1.0).unwrap();

    // reshaped.plot("ml_rs.png");

    // if let Ok(ds) = reshaped.generate_aggregate(0.2) {
    //     ds.plot(true, "ml_agg2.png");
    // }

    // for ds in reshaped.datasets.iter_mut() {
    //     if let Some(last) = ds.series.last() {
    //         if let Ok(hdd) = ds.clone().extrapolate_higher_deriv(vec![0.1, 0.1], 2, 1.0, last.0 + 2.0) {
    //             *ds = hdd;
    //         }
    //     }
    // }

    // std::fs::write("trainer.json", serde_json::to_string_pretty(&reshaped).unwrap());

    // return;

    let settings = load_settings();

    let peeper = Arc::new(Mutex::new(PixelPeeper {
        buffers: Vec::new(),
    }));
    let peeper_c = peeper.clone();
    // std::thread::spawn(move || {
    //     PixelPeeper::peeper_daemon(peeper_c);
    // });

    let sm = Arc::new(SmartTrigger::new(motion::MotionDetector::new(
        settings.dynamic_rect[0],
        settings.dynamic_rect[1],
        settings.dynamic_rect[2],
        settings.dynamic_rect[3],
    )));
    let smart_trigger = TriggerRes(sm.clone());

    std::thread::spawn(move || {
        SmartTrigger::daemon(sm);
    });

    let mut plate_detectors = settings
        .plate_rects
        .iter()
        .map(|(r, pos)| {
            let sm = Arc::new(SmartTrigger::new(motion::MotionDetector::new_green(
                r[0], r[1], r[2], r[3],
            )));
            let sm_cl = sm.clone();
            std::thread::spawn(move || {
                SmartTrigger::daemon(sm_cl);
            });
            (sm, *pos)
        })
        .collect();

    let board_trigger = BoardTriggerRes(plate_detectors);

    let history_res = {
        if let Some(x) = std::fs::read_to_string("history.json")
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
        {
            x
        } else {
            HistoryRes(Vec::new())
        }
    };

    App::new()
        .add_event::<GlobalClickEvent>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: WindowResolution::new(650., 700.).with_scale_factor_override(1.0),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(OverlayPlugin {
            font_size: 18.0,
            ..default()
        })
        .add_systems(Startup, create_board)
        .add_systems(
            Update,
            (
                input,
                update_plate,
                update_dynamic,
                autopilot_system,
                auto_trigger,
            ),
        )
        .add_event::<Reset>()
        .add_event::<Train>()
        .insert_resource(PeeperRes(peeper))
        .insert_resource(smart_trigger)
        .insert_resource(board_trigger)
        .insert_resource(MlRes::new(trainer, settings.weights.clone()))
        .insert_resource(ApTrainRes { train: false })
        .insert_resource(settings)
        .insert_resource(history_res)
        .insert_resource(AutopilotState::Offline)
        .run();
}

#[derive(Component)]
pub struct Plate;

#[derive(Component)]
pub struct Dynamic(Option<f64>);

#[derive(Resource)]
pub struct Sim {
    state: SimState,
    plate: Option<PlateState>,
    dynamic: Option<DynamicState>,
    clockwise: bool,
    solution: Option<Solution>,
    bet: Option<Bet>,
    train_dyn: bool,
    recorded_final_vel: Option<f64>,
}

#[derive(Resource)]
pub struct MlRes {
    pub trainer: MlTrainer,
    pub ticker: Option<Ticker>,
    pub weights: Vec<MlWeights>,
    pub prediction: Option<(Prediction, SystemTime)>,
    pub min_tick_t: f64,
}

impl MlRes {
    pub fn new(trainer: MlTrainer, weights: Vec<MlWeights>) -> Self {
        Self {
            trainer,
            ticker: None,
            prediction: None,
            min_tick_t: 15.0,
            weights,
        }
    }
}

#[derive(Resource)]
pub struct Ticker {
    pub start: SystemTime,
    pub ticks: Vec<f64>,
}

impl Ticker {
    pub fn start() -> Self {
        Self {
            start: SystemTime::now(),
            ticks: Vec::new(),
        }
    }

    pub fn series(&self) -> Series {
        self.ticks
            .iter()
            .enumerate()
            .map(|(i, t)| (*t, -std::f64::consts::PI * 2.0 * i as f64))
            .collect()
    }

    pub fn cleaned_series(&self, max_t: f64) -> Option<Series> {
        if self.ticks.len() < 2 {
            return None;
        }
        let first = *self.ticks.first()?;

        let series: Vec<(f64, f64)> = self
            .ticks
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                let tafter = t - first;
                if tafter > max_t {
                    None
                } else {
                    let x = -std::f64::consts::PI * 2.0 * i as f64;
                    Some((tafter, x))
                }
            })
            .collect();

        if series.len() < 2 {
            return None;
        }

        let deltas: Vec<f64> = (0..series.len() - 1)
            .map(|i| series[i + 1].0 - series[i].0)
            .collect();
        for i in 0..deltas.len() - 1 {
            if deltas[i + 1] < deltas[i] {
                return None;
            }
        }

        Some(series)
    }
}

#[derive(Debug, Resource)]
pub struct ApTrainRes {
    pub train: bool,
}

#[derive(Resource, Debug)]
pub enum AutopilotState {
    InTrigger,
    AwaitingDrop,
    AwaitingResult {
        viewer: Arc<Mutex<NumberViewer>>,
        overtime: bool,
    },
    Standby,
    AwaitingReset,
    Offline,
}

impl std::fmt::Display for AutopilotState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InTrigger => f.write_str("InTrigger"),
            Self::AwaitingDrop => f.write_str("AwaitingDrop"),
            Self::Standby => f.write_str("Standby"),
            Self::AwaitingReset => f.write_str("AwaitingReset"),
            Self::Offline => f.write_str("Offline"),
            Self::AwaitingResult { overtime, .. } => {
                f.write_fmt(format_args!("AwaitingResult(overtime={})", overtime))
            }
        }
    }
}

#[derive(Resource)]
pub struct TriggerRes(Arc<SmartTrigger>);

#[derive(Resource)]
pub struct BoardTriggerRes(Vec<(Arc<SmartTrigger>, f64)>);

const R: f32 = 270.0;

const R_1: f32 = 150.0;

const DYNAMIC_THRESHOLD: f32 = 0.55;
const PLATE_THRESHOLD: f32 = 0.08;

const CLICK_POS: f64 = 0.0;

//const DYNAMIC_COL: Color = Color::rgb(0.9, 0.9, 0.9);
const DYNAMIC_COL: Color = Color::rgb(0.694, 0.694, 0.776);
//const PLATE_COL: Color = Color::rgb(0.2, 0.45, 0.2);
const PLATE_COL: Color = Color::rgb(0.4196, 0.709, 0.4196);

const MIN_INTERVAL_DYNAMIC: f64 = 0.3;
const MIN_INTERVAL_PLATE: f64 = 0.3;
const PLATE_TRIGGER_0: f64 = 0.0;
const PLATE_TRIGGER_1: f64 = std::f64::consts::FRAC_PI_2;
const PLATE_TRIGGER_2: f64 = std::f64::consts::PI;
const PLATE_TRIGGER_3: f64 = std::f64::consts::FRAC_PI_2 * 3.0;

fn create_board(mut commands: Commands, asset_server: Res<AssetServer>, settings: Res<Settings>) {
    commands.spawn(Camera2dBundle::default());
    commands.spawn((
        Plate,
        SpriteBundle {
            texture: asset_server.load("plate.png"),
            ..default()
        },
    ));
    commands.spawn((
        Dynamic(None),
        SpriteBundle {
            texture: asset_server.load("dynamic.png"),
            transform: Transform::from_xyz(0.0, R, 0.0).with_scale(Vec3::new(0.02, 0.02, 0.02)),
            ..default()
        },
    ));
    commands.insert_resource(Sim {
        state: new_state(true, &settings),
        plate: None,
        dynamic: None,
        clockwise: true,
        solution: None,
        bet: None,
        recorded_final_vel: None,
        train_dyn: false,
    });
    commands.insert_resource(AutoTrigger {
        dynamic_points: Vec::new(),
        plate_points: Vec::new(),
        state: TriggerState::Idle,
        prev_down: false,
        start: None,
    });
    commands.insert_resource(ApTrainRes { train: false });

    let trainer = if let Ok(s) = std::fs::read_to_string("trainer.json") {
        serde_json::from_str(&s).unwrap_or_default()
    } else {
        MlTrainer::default()
    };

    commands.insert_resource(MlRes::new(trainer, settings.weights.clone()));
    commands.insert_resource(AutopilotState::Offline);
}

fn new_state(clockwise: bool, settings: &Settings) -> SimState {
    SimState::new(
        clockwise,
        SystemTime::now(),
        0.01,
        SimParams {
            plate_acc: settings.plate_acc,
            dynamic_acc: settings.dynamic_acc,
            min_vel: settings.min_vel,
            k: settings.k,
            att: settings.att,
            dynamic_weights: SVector::from(settings.dynamic_weights),
            end_t: settings.end_t,
            end_d: settings.end_d,
        },
    )
}

// fn print_solve(state: &SimState) {
//     let sln = state.solve();
//     if let Some(sln) = sln {
//         let slot = state
//             .plate_state
//             .unwrap()
//             .slot_at_local_pos(sln, LAYOUT.len());
//         screen_print!(sec: 20.0, col: Color::CYAN, "current sln: {}, i: {}, n: {}", sln, slot, LAYOUT[slot]);
//     }
// }

fn input(
    mut commands: Commands,
    mut reset_events: EventReader<Reset>,
    mut train_events: EventReader<Train>,
    mut key_evr: EventReader<KeyboardInput>,
    mut sim: ResMut<Sim>,
    mut settings: ResMut<Settings>,
    mut trigger: ResMut<AutoTrigger>,
    mut peeper: ResMut<PeeperRes>,
    mut smart_trigger: ResMut<TriggerRes>,
    mut board_trigger: ResMut<BoardTriggerRes>,
    mut ap: ResMut<AutopilotState>,
    mut ml: ResMut<MlRes>,
    mut ap_train: ResMut<ApTrainRes>,
    mut set: ParamSet<(
        Query<(&Plate, &mut Transform)>,
        Query<(&mut Dynamic, &mut Transform)>,
    )>,
) {
    let mut reset: bool = false;
    for ev in key_evr.iter() {
        if let ButtonState::Pressed = ev.state {
            match ev.key_code {
                Some(KeyCode::P) => {
                    sim.state.plate_click(SystemTime::now(), CLICK_POS, 0.3);
                    sim.plate = sim.state.plate_state;
                }
                Some(KeyCode::D) => {
                    sim.state
                        .dynamic_click(SystemTime::now(), CLICK_POS, 0.4, 0.3, 0.7, 0.3);
                    sim.dynamic = sim.state.dynamic_state;
                }
                Some(KeyCode::C) => {
                    sim.clockwise = !sim.clockwise;
                    screen_print!(sec: 5.0, col: Color::ORANGE, "clockwise = {}", sim.clockwise);
                    if sim.plate.is_none() && sim.dynamic.is_none() {
                        reset = true;
                    }
                }
                Some(KeyCode::T) => match trigger.state {
                    TriggerState::Idle => {
                        trigger.state = TriggerState::Setup(0);
                        trigger.reset();
                        screen_print!(sec: 5.0, col: Color::YELLOW, "TRIGGER SETUP");
                    }
                    TriggerState::Setup(_) => {
                        trigger.state = TriggerState::Idle;
                        screen_print!(sec: 5.0, col: Color::YELLOW_GREEN, "TRIGGER IDLE");
                    }
                    _ => {}
                },
                Some(KeyCode::U) => {
                    sim.train_dyn = !sim.train_dyn;
                    screen_print!(sec: 5.0, col: Color::YELLOW, "TRAIN DYNAMIC MODE: {}", sim.train_dyn);
                }
                Some(KeyCode::Y) => match trigger.state {
                    TriggerState::Idle => {
                        trigger.state = TriggerState::PatternSetup;
                        trigger.reset();
                        screen_print!(sec: 5.0, col: Color::YELLOW, "TRIGGER PATTERN SETUP");
                    }
                    TriggerState::PatternSetup => {
                        trigger.state = TriggerState::Idle;
                        screen_print!(sec: 5.0, col: Color::YELLOW_GREEN, "TRIGGER IDLE");
                    }
                    _ => {}
                },
                Some(KeyCode::A) => {
                    match trigger.state {
                        TriggerState::Idle => {
                            // if trigger.is_ready() {
                            //     trigger.state = TriggerState::Armed;
                            //     smart_trigger.0.flush_detect(Duration::from_millis(50)); // Flush so we don't get a detection at the start.
                            //     screen_print!(sec: 5.0, col: Color::RED, "TRIGGER ARMED");
                            // } else {
                            //     screen_print!(sec: 5.0, col: Color::YELLOW, "could not arm trigger - not enough points specified");
                            // }

                            trigger.state = TriggerState::Armed;
                            smart_trigger.0.clear(); // Flush so we don't get a detection at the start.
                                                     // Start the ticker if trainimg mode.
                            if sim.train_dyn {
                                ml.ticker = Some(Ticker::start())
                            }
                            for (tr, _) in board_trigger.0.iter() {
                                tr.clear();
                            }
                            screen_print!(sec: 5.0, col: Color::RED, "TRIGGER ARMED");
                        }
                        TriggerState::Armed => {
                            trigger.state = TriggerState::Idle;
                            screen_print!(sec: 5.0, col: Color::YELLOW_GREEN, "TRIGGER IDLE");
                        }
                        TriggerState::Setup(n) => {
                            trigger.reset();
                            if n >= settings.dynamic_points.len() + settings.plate_points.len() {
                                trigger.state = TriggerState::Armed;
                                screen_print!(sec: 5.0, col: Color::RED, "TRIGGER ARMED");
                            } else {
                                screen_print!(sec: 5.0, col: Color::YELLOW, "could not arm trigger - not enough points specified");
                            }
                        }
                        _ => {}
                    }
                }
                Some(KeyCode::S) => {
                    if let TriggerState::Active = trigger.state {
                        trigger.state = TriggerState::Idle;
                        screen_print!(sec: 5.0, col: Color::YELLOW_GREEN, "TRIGGER IDLE");
                        if let Some(sln) = sim.state.solve() {
                            sim.solution = Some(sln);
                            screen_print!(sec: 20.0, col: Color::CYAN, "sln: {}, i: {}, n: {}", sln.plate_pos, sln.i, sln.n);
                        }
                    }
                }
                Some(KeyCode::V) => {
                    if let TriggerState::Active = trigger.state {
                        if let Some((pr, t)) = &ml.prediction {
                            let delta = SystemTime::now().duration_since(*t).unwrap().as_secs_f64();

                            let tx = pr.t_head + delta;
                            screen_print!(sec: 5.0, col: Color::PINK, "TIME (tx): {}", tx);
                        }
                        if let Some(dy) = sim.dynamic {
                            screen_print!(sec: 5.0, col: Color::PINK, "VELOCITY: {}", dy.vel);
                            sim.recorded_final_vel = Some(dy.vel.abs());
                        }
                    }
                }
                Some(KeyCode::X) => {
                    *ap = AutopilotState::Standby;
                    screen_print!(sec: 5.0, col: Color::ORANGE, "AUTOPILOT ENGAGED");
                }
                Some(KeyCode::L) => {
                    // Train the model.
                    screen_print!(sec: 5.0, col: Color::BLUE, "COMMENCED LEARNING ({} clicks)", sim.state.dynamic_clicks.len());

                    // ML learning
                    if let Some(ticker) = &ml.ticker {
                        let series = ticker.series();
                        println!("Series: {:?}", &series);
                        if let Err(e) = ml.trainer.add_series_aligned(series, 1.0) {
                            screen_print!(sec: 5.0, col: Color::RED, "ERROR: Failed to perform ML learning: {}", e);
                        } else {
                            if let Ok(s) = serde_json::to_string_pretty(&ml.trainer) {
                                let _ = std::fs::write("trainer.json", &s);
                            }
                            ml.trainer.plot("ml.png");
                        }
                    }

                    let old_k = sim.state.params.k;
                    let old_att = sim.state.params.att;
                    let outcome = sim.state.train(1.0, 1000, 0.4, 0.01, 0.05);

                    if let Some(outcome) = outcome {
                        screen_print!(sec: 5.0, col: Color::BLUE, "LEARNING COMPLETE");
                        sim.state.plot(&sim.state.params, "before.png");
                        sim.state.params = outcome;
                        sim.state.plot(&sim.state.params, "after.png");
                        settings.write_params("settings.json", outcome, sim.recorded_final_vel);
                    } else {
                        screen_print!(sec: 5.0, col: Color::RED, "LEARNING FAILED");
                    }
                }
                Some(KeyCode::M) => {
                    // Train the model.

                    ap_train.train = !ap_train.train;
                    screen_print!(sec: 5.0, col: Color::PINK, "AUTOPILOT TRAINING ENABLED: {}", ap_train.train);
                }
                Some(KeyCode::R) => {
                    reset = true;
                }
                _ => {}
            }
        }
    }

    if train_events.read().count() > 0 {
        // Train
        if let Some(ticker) = ml.ticker.take() {
            let max_t = if let Some((pr, t)) = &ml.prediction {
                if let Some(t0) = pr.head.first().map(|x| x.0) {
                    (pr.t_end - t0) + 2.5
                } else {
                    ml.min_tick_t
                }
            } else {
                ml.min_tick_t
            };

            if let Some(series) = ticker.cleaned_series(max_t) {
                // create weights
                if let Some(weight) = ml.trainer.weights(series.clone(), 10) {
                    let len = weight.0.len();
                    ml.weights.push(weight);
                    while ml.weights.len() > 6 {
                        ml.weights.remove(0); // Remove element added the longest time ago.
                    }
                    screen_print!(sec: 5.0, col: Color::PINK, "ADDED WEIGHTS (ticks = {}, n = {}, weights={})", ticker.ticks.len(), len, ml.weights.len());
                    settings.weights = ml.weights.clone();

                    settings.write("settings.json");
                }

                if let Some(last) = series.last().cloned() {
                    if let Ok(hdd) = Dataset::new_aligned(series, ml.trainer.v_star) {
                        if let Ok(hdd) =
                            hdd.extrapolate_higher_deriv(vec![0.1, 0.1], 2, 1.0, last.0 + 2.0)
                        {
                            if let Err(e) = ml.trainer.add_series_aligned(hdd.series, 1.0) {
                                screen_print!(sec: 5.0, col: Color::RED, "TRAINING FAILED: {}", e);
                            } else {
                                if let Ok(s) = serde_json::to_string_pretty(&ml.trainer) {
                                    let _ = std::fs::write("trainer.json", &s);
                                }
                                ml.trainer.plot("ml.png");
                                screen_print!(sec: 5.0, col: Color::PINK, "TRAINING SUCCEEDED");
                            }
                        } else {
                            screen_print!(sec: 5.0, col: Color::RED, "TRAINING FAILED - EXTRAPOLATION ERROR");
                        }
                    }
                }
            } else {
                screen_print!(sec: 5.0, col: Color::RED, "TRAINING FAILED - UNCLEAN");
            }
        }
    }

    if reset || reset_events.read().count() > 0 {
        *settings = load_settings();

        commands.insert_resource(Sim {
            state: new_state(sim.clockwise, &settings),
            plate: None,
            dynamic: None,
            clockwise: sim.clockwise,
            solution: None,
            bet: None,
            recorded_final_vel: None,
            train_dyn: false,
        });

        sim.state = new_state(sim.clockwise, &settings);
        sim.plate = None;
        sim.dynamic = None;
        sim.solution = None;
        sim.bet = None;
        sim.recorded_final_vel = None;
        sim.train_dyn = false;

        ml.ticker = None;
        ml.prediction = None;

        if reset {
            *ap = AutopilotState::Offline;
        }

        for (plate, mut transform) in set.p0().iter_mut() {
            transform.translation = Vec3::ZERO;
            transform.rotation = Quat::IDENTITY;
        }

        for (mut dynamic, mut transform) in set.p1().iter_mut() {
            transform.translation = Vec3::new(0.0, R, 0.0);
            transform.rotation = Quat::IDENTITY;
            dynamic.0 = None;
        }

        trigger.state = TriggerState::Idle;

        peeper.0.lock().unwrap().clear();

        screen_print!(col: Color::BLACK, "RESET");
    }
}

pub fn autopilot_system(
    mut commands: Commands,
    mut ap: ResMut<AutopilotState>,
    mut sim: ResMut<Sim>,
    mut settings: ResMut<Settings>,
    mut trigger: ResMut<AutoTrigger>,
    mut history: ResMut<HistoryRes>,
    mut smart_trigger: ResMut<TriggerRes>,
    mut board_trigger: ResMut<BoardTriggerRes>,
    mut reset_events: EventWriter<Reset>,
    mut train_events: EventWriter<Train>,
    mut ml: ResMut<MlRes>,
    mut ap_train: ResMut<ApTrainRes>,
    mut set: ParamSet<(
        Query<(&Plate, &mut Transform)>,
        Query<(&mut Dynamic, &mut Transform)>,
    )>,
) {
    let mut next_state = None;
    match ap.deref() {
        AutopilotState::Standby => {
            // wait for dark rect
            if let Some(col) = average_color(settings.autopilot_rect) {
                let tr_col = Color::rgb_u8(
                    settings.autopilot_trigger_col[0],
                    settings.autopilot_trigger_col[1],
                    settings.autopilot_trigger_col[2],
                );

                let tr_vec = Vec4::from_array(tr_col.as_rgba_f32());
                let col_vec = Vec4::from_array(col.as_rgba_f32());

                if tr_vec.distance(col_vec) < settings.autopilot_sensitivity as f32 {
                    // Enter trigger state, reset any values.
                    next_state = Some(AutopilotState::InTrigger);
                    trigger.state = TriggerState::Armed;
                    std::thread::sleep(Duration::from_millis(10));
                    smart_trigger.0.clear(); // Flush so we don't get a detection at the start.
                    ml.prediction = None;
                    ml.ticker = Some(Ticker::start());
                    for (tr, _) in board_trigger.0.iter() {
                        tr.clear();
                    }
                }
            }
        }
        AutopilotState::InTrigger => {
            // wait for solution or timeout.
            if let Some(start) = trigger.start {
                if SystemTime::now()
                    .duration_since(start)
                    .map(|x| x > Duration::from_secs_f64(15.0))
                    .unwrap_or(true)
                {
                    screen_print!(sec: 5.0, col: Color::ORANGE, "Timout waiting for trigger start");
                    trigger.state = TriggerState::Idle;
                    next_state = Some(AutopilotState::AwaitingReset);
                }
            }
            // if let Some(sln) = sim.solution {
            //     if let Some(dy) = sim.dynamic {
            //         if dy.vel < settings.bet_vel {
            //             // Bet on solution and disarm trigger.
            //             screen_print!(sec: 20.0, col: Color::CYAN, "BETTING ON: sln: {}, i: {}, n: {}", sln.plate_pos, sln.i, sln.n);
            //             let bet = Bet::surrounding(sln.i, 10);
            //             sim.bet = Some(bet);
            //             next_state = Some(AutopilotState::AwaitingDrop);
            //         }
            //     }
            // }

            let mut solved = false;
            if let Some((pr, start)) = &ml.prediction {
                let ps_t_end = sim
                    .state
                    .time(*start + Duration::from_secs_f64(pr.t_end - pr.t_head));
                if let Some(sln) = sim.state.finalize(ps_t_end, pr.x_end) {
                    screen_print!(sec: 20.0, col: Color::CYAN, "sln: {}, i: {}, n: {}", sln.plate_pos, sln.i, sln.n);
                    sim.solution = Some(sln);
                    let bet = Bet::surrounding(sln.i, 10);
                    sim.bet = Some(bet);
                    next_state = Some(AutopilotState::AwaitingDrop);
                } // else {
                  //     screen_print!(sec: 2.0, col: Color::RED, "ERROR finding solution");
                  //     trigger.state = TriggerState::Idle;
                  //     next_state = Some(AutopilotState::AwaitingReset);
                  // }
            }
        }
        AutopilotState::AwaitingDrop => {
            if let Some(start) = trigger.start {
                if SystemTime::now()
                    .duration_since(start)
                    .map(|x| x < Duration::from_secs_f64(20.0))
                    .unwrap_or(true)
                {
                    screen_print!(sec: 5.0, col: Color::ORANGE, "Timout waiting for drop");
                    trigger.state = TriggerState::Idle;
                    next_state = Some(AutopilotState::AwaitingReset);
                }
            }

            if let Some((pr, start)) = &ml.prediction {
                let t = SystemTime::now()
                    .duration_since(*start)
                    .unwrap()
                    .as_secs_f64()
                    + pr.t_head;
                if t > pr.t_end {
                    let viewer = Arc::new(Mutex::new(NumberViewer::new(settings.result_rect)));
                    let viewer_cl = viewer.clone();
                    std::thread::spawn(move || {
                        NumberViewer::daemon(viewer_cl);
                    });
                    next_state = Some(AutopilotState::AwaitingResult {
                        viewer,
                        overtime: false,
                    });
                }
            }
        }
        AutopilotState::AwaitingResult { viewer, overtime } => {
            if let Ok(mut v) = viewer.try_lock() {
                if v.changed() {
                    // Store the result
                    let num = v.number();
                    if let Some(num) = num {
                        if let Some(sln) = sim.solution {
                            let outcome = sln.create_outcome(num as u8);
                            if let Some(outcome) = outcome {
                                history.0.push(SimRecord {
                                    sln,
                                    outcome,
                                    bet: sim.bet.clone(),
                                });
                                if let Some(bet) = &sim.bet {
                                    screen_print!(sec: 7.0, col: Color::WHITE, "OUTCOME: {}, bet succeeded: {}", outcome.n, bet.0.contains(&outcome.n));
                                } else {
                                    screen_print!(sec: 7.0, col: Color::WHITE, "OUTCOME: {}", outcome.n);
                                }

                                if let Ok(c) = serde_json::to_string_pretty(history.deref()) {
                                    std::fs::write("history.json", c);
                                }
                                println!(
                                    "Expected Returns: {:?}, Winrate: {:?}, n: {}",
                                    history.constant_bet_returns(),
                                    history.winrate(),
                                    history.0.len(),
                                );
                            } else {
                                screen_print!(sec: 5.0, col: Color::RED, "ERROR: Failed to generate outome!");
                            }
                        } else {
                            screen_print!(sec: 5.0, col: Color::RED, "ERROR: No solution!");
                        }
                    }
                    v.drop = true;
                    next_state = Some(AutopilotState::AwaitingReset);
                    trigger.state = TriggerState::Idle;
                    if ap_train.train {
                        train_events.send(Train);
                    }
                    reset_events.send(Reset);
                }
            }

            if *overtime {
                if let Some(col) = average_color(settings.autopilot_rect) {
                    let tr_col = Color::rgb_u8(
                        settings.autopilot_trigger_col[0],
                        settings.autopilot_trigger_col[1],
                        settings.autopilot_trigger_col[2],
                    );

                    let tr_vec = Vec4::from_array(tr_col.as_rgba_f32());
                    let col_vec = Vec4::from_array(col.as_rgba_f32());

                    // New spin, so the number must be the same as last time, hence why it has not changed.
                    if tr_vec.distance(col_vec) < settings.autopilot_sensitivity as f32 {
                        // Store existing result.
                        let num = viewer.lock().unwrap().number();
                        screen_print!(sec: 5.0, "Using same number due to overtime ({:?})", num);
                        if let Some(num) = num {
                            if let Some(sln) = sim.solution {
                                let outcome = sln.create_outcome(num as u8);
                                if let Some(outcome) = outcome {
                                    history.0.push(SimRecord {
                                        sln,
                                        outcome,
                                        bet: sim.bet.clone(),
                                    });
                                    if let Ok(c) = serde_json::to_string(history.deref()) {
                                        std::fs::write("history.json", c);
                                    }

                                    if let Some(bet) = &sim.bet {
                                        screen_print!(sec: 7.0, col: Color::WHITE, "OUTCOME: {}, bet succeeded: {}", outcome.n, bet.0.contains(&outcome.n));
                                    } else {
                                        screen_print!(sec: 7.0, col: Color::WHITE, "OUTCOME: {}", outcome.n);
                                    }

                                    println!(
                                        "Expected Returns: {:?}, Winrate: {:?}, obs: {}",
                                        history.constant_bet_returns(),
                                        history.winrate(),
                                        history.0.len(),
                                    );
                                } else {
                                    screen_print!(sec: 5.0, col: Color::RED, "ERROR: Failed to generate outome!");
                                }
                            } else {
                                screen_print!(sec: 5.0, col: Color::RED, "ERROR: No solution!");
                            }
                        }

                        viewer.lock().unwrap().drop = true;
                        next_state = Some(AutopilotState::Standby);
                        trigger.state = TriggerState::Idle;
                        if ap_train.train {
                            train_events.send(Train);
                        }
                        reset_events.send(Reset);
                    }
                }
            } else {
                if let Some(col) = average_color(settings.autopilot_rect) {
                    let tr_col = Color::rgb_u8(
                        settings.autopilot_stby_col[0],
                        settings.autopilot_stby_col[1],
                        settings.autopilot_stby_col[2],
                    );

                    let tr_vec = Vec4::from_array(tr_col.as_rgba_f32());
                    let col_vec = Vec4::from_array(col.as_rgba_f32());

                    // New spin, so the number must be the same as last time, hence why it has not changed.
                    if tr_vec.distance(col_vec) < settings.autopilot_sensitivity as f32 {
                        next_state = Some(AutopilotState::AwaitingResult {
                            viewer: viewer.clone(),
                            overtime: true,
                        });
                    }
                }
            }
        }
        AutopilotState::AwaitingReset => {
            if let Some(col) = average_color(settings.autopilot_rect) {
                let tr_col = Color::rgb_u8(
                    settings.autopilot_stby_col[0],
                    settings.autopilot_stby_col[1],
                    settings.autopilot_stby_col[2],
                );

                let tr_vec = Vec4::from_array(tr_col.as_rgba_f32());
                let col_vec = Vec4::from_array(col.as_rgba_f32());
                if tr_vec.distance(col_vec) < settings.autopilot_sensitivity as f32 {
                    // Enter trigger state.
                    next_state = Some(AutopilotState::Standby);
                    if ap_train.train {
                        train_events.send(Train);
                    }
                    reset_events.send(Reset);
                }
            }
        }
        AutopilotState::Offline => {}
    }

    if let Some(next_state) = next_state {
        screen_print!(sec: 5.0, col: Color::ORANGE, "AUTOPILOT STATE: {}", &next_state);
        if let AutopilotState::Standby = next_state {
            heartbeat_interaction(settings.heartbeat_coords, settings.undo_coords);
        }
        *ap = next_state;
    }
}

pub fn update_plate(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut plates: Query<(&Plate, &mut Transform)>,
) {
    for (plate, mut transform) in plates.iter_mut() {
        let ps = sim.plate;
        if let Some(ps) = ps {
            let next = ps.approximate(time.delta_seconds() as f64, sim.state.step);

            transform.rotation = Quat::from_rotation_z(-next.dis as f32);
            sim.plate = Some(next);
        }
    }
}

#[derive(Event)]
pub struct Reset;

#[derive(Event)]
pub struct Train;

pub fn update_dynamic(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    ml: Res<MlRes>,
    mut dynamics: Query<(&mut Dynamic, &mut Transform)>,
) {
    for (mut dynamic, mut transform) in dynamics.iter_mut() {
        if let Some((pr, start)) = &ml.prediction {
            let t = SystemTime::now()
                .duration_since(*start)
                .unwrap()
                .as_secs_f64()
                + pr.t_head;
            if t > pr.t_end {
                if let Some(plate) = sim.plate {
                    if let Some(sln) = &sim.solution {
                        let p = plate.dis + sln.plate_pos;
                        transform.translation.x = R_1 * p.sin() as f32;
                        transform.translation.y = R_1 * p.cos() as f32;
                    }
                }
            } else {
                let dis = pr.zero_dis(t);
                if let Some(dis) = dis {
                    transform.translation.x = R * dis.sin() as f32;
                    transform.translation.y = R * dis.cos() as f32;
                }
            }
        } else {
            if let Some(f) = dynamic.0 {
                if let Some(plate) = sim.plate {
                    let p = plate.dis + f;
                    transform.translation.x = R_1 * p.sin() as f32;
                    transform.translation.y = R_1 * p.cos() as f32;
                }
            } else {
                let ds = sim.dynamic;
                if let Some(ds) = ds {
                    let next = ds.approximate(
                        time.delta_seconds() as f64,
                        sim.state.step,
                        sim.state.params.dynamic_acc,
                        sim.state.params.k,
                        sim.state.params.att,
                        sim.state.params.dynamic_weights,
                    );
                    transform.translation.x = R * next.dis.sin() as f32;
                    transform.translation.y = R * next.dis.cos() as f32;

                    sim.dynamic = Some(next);

                    if ds.vel.abs() < sim.state.params.min_vel && !sim.train_dyn {
                        //beep();
                        if let Some(sln) = sim.state.finalize(ds.t, ds.dis) {
                            //screen_print!(sec: 20.0, col: Color::DARK_GREEN, "SOLUTION: sln: {}, i: {}, n: {}", sln.plate_pos, sln.i, sln.n);
                            dynamic.0 = Some(sln.plate_pos)
                        }

                        sim.dynamic = None;
                    }
                }
            }
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct TriggerPoint {
    points: Vec<autopilot::geometry::Point>,
    pos: f64,
    prev: Option<SystemTime>,
}

pub enum TriggerState {
    Idle,
    Setup(usize),
    PatternSetup,
    Armed,
    Active,
}

#[derive(Resource)]
pub struct AutoTrigger {
    dynamic_points: Vec<TriggerPoint>,
    plate_points: Vec<TriggerPoint>,

    state: TriggerState,
    prev_down: bool,

    start: Option<SystemTime>,
}

impl AutoTrigger {
    pub fn is_ready(&self) -> bool {
        !self.dynamic_points.is_empty() && !self.plate_points.is_empty()
    }

    pub fn reset(&mut self) {
        self.dynamic_points.clear();
        self.plate_points.clear();
    }
}

// fn bot_system(mut trigger: ResMut<AutoTrigger>, mut number_viewer: ResMut<NumberViewer>) {

// }

fn auto_trigger(
    mut trigger: ResMut<AutoTrigger>,
    mut sim: ResMut<Sim>,
    mut peeper: ResMut<PeeperRes>,
    mut smart_trigger: ResMut<TriggerRes>,
    mut board_trigger: ResMut<BoardTriggerRes>,
    mut ml: ResMut<MlRes>,
    settings: Res<Settings>,
) {
    use device_query::MouseState;
    use device_query::{DeviceEvents, DeviceQuery, DeviceState};

    let device_state = DeviceState::new();
    let mouse: MouseState = device_state.get_mouse();

    let click: Option<Vec2> = if mouse.button_pressed[1] == true && !trigger.prev_down {
        Some(Vec2::new(mouse.coords.0 as f32, mouse.coords.1 as f32))
    } else {
        None
    };

    trigger.prev_down = mouse.button_pressed[1];

    match trigger.state {
        TriggerState::Active | TriggerState::Armed => {
            if let Some((r, t)) = smart_trigger.0.flush_detect(Duration::from_secs_f64(0.2)) {
                if SystemTime::now().duration_since(t).unwrap() < Duration::from_secs_f64(0.2) {
                    screen_print!(sec: 0.2, col: Color::WHITE, "smart trigger click");
                    //println!("smart trigger");

                    // if ml.ticker.is_none() {
                    //     let cap_width = settings.dynamic_rect[2];

                    //     let p = r.center().x as f64;

                    //     let x = settings.dynamic_rect_pos + p;

                    //     let r = settings.board_rad;

                    //     let a = (x / r).acos();

                    //     let c = settings.dynamic_rect_pos + (settings.dynamic_rect[3] / 2) as f64;

                    //     let ac = (c / r).acos();

                    //     let rel_a = a - ac;

                    //     //println!("TR: {}", pos);

                    //     if sim.train_dyn {
                    //         sim.state.dynamic_click(t, rel_a, 0.2, 0.2, 1.0, 1.0);
                    //     } else {
                    //         sim.state.dynamic_click(t, rel_a, 0.6, 0.5, 0.8, 0.8);
                    //     }

                    //     sim.dynamic = sim.state.dynamic_state;

                    //     //Solve if vel is correct
                    //     if let Some(dy) = sim.dynamic {
                    //         if dy.vel < settings.solve_vel {
                    //             if let Some(sln) = sim.state.solve() {
                    //                 sim.solution = Some(sln);
                    //                 screen_print!(sec: 20.0, col: Color::CYAN, "sln: {}, i: {}, n: {}", sln.plate_pos, sln.i, sln.n);
                    //             } else {
                    //                 screen_print!(sec: 2.0, col: Color::RED, "ERROR finding solution");
                    //             }
                    //         }
                    //     }
                    // }

                    let mut start = None;

                    let pr_is_none = ml.prediction.is_none();

                    if let Some(ticker) = &mut ml.ticker {
                        let now = t;
                        let tx = now.duration_since(ticker.start).unwrap().as_secs_f64();
                        ticker.ticks.push(tx);
                        if pr_is_none && ticker.ticks.len() >= 3 {
                            let t0 = ticker.ticks[ticker.ticks.len() - 2];

                            if tx - t0 > 1.0 {
                                start = Some(now);
                            }
                        }

                        //if ticker.
                    }

                    if let Some(start) = start {
                        let series = ml.ticker.as_ref().unwrap().series();

                        let agg = if ml.weights.len() >= 2 {
                            ml.trainer.generate_weighted(0.2, &ml.weights)
                        } else {
                            ml.trainer.generate_aggregate(0.2)
                        };

                        if let Ok(ds) = agg {
                            let te = ds.series.last().unwrap().0;
                            if let Ok(eds) =
                                ds.extrapolate_higher_deriv(vec![0.1, 0.1], 6, 1.0, 12.0)
                            {
                                match eds.predict(series, 8.0, 0.0) {
                                    Ok(pr) => {
                                        println!("te: {}, t_end: {}", te, pr.t_end);
                                        Dataset::plot_series(
                                            &[&pr.path.series, &pr.head],
                                            "sln.png",
                                        );
                                        // Calculate solution!!
                                        ml.prediction = Some((pr, start));
                                    }
                                    Err(e) => {
                                        screen_print!(col: Color::RED, "Failed to predict: {}", e)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // for (i, dp) in trigger.dynamic_points.iter_mut().enumerate() {
            //     if let Some(t) = peeper.0.lock().unwrap().flush_or_add(&dp.points, Color::rgb(settings.dynamic_color[0], settings.dynamic_color[1], settings.dynamic_color[2]), settings.dynamic_threshold) {

            //         // let cvec = DYNAMIC_COL;

            //         // let tvec = Vec3::new(col.r(), col.g(), col.b());

            //         // screen_print!("dynamic trigger color: {}, {}, {}", tvec.x, tvec.y, tvec.z);

            //         // let diff = cvec - tvec;

            //         let time_since_last = dp.prev.map(|x| t.duration_since(x).unwrap().as_secs_f64());

            //         if time_since_last.unwrap_or(MIN_INTERVAL_DYNAMIC) >= MIN_INTERVAL_DYNAMIC {
            //             // Trigger.
            //             screen_print!(sec: 0.5, "dynamic trigger click, i={}", i);

            //             sim.state.dynamic_click(t, dp.pos, 1.0);
            //             sim.dynamic = sim.state.dynamic_state;
            //             dp.prev = Some(t);
            //         }
            //     }

            // }
            for (tr, pos) in board_trigger.0.iter() {
                if let Some((r, t)) = tr.flush_detect(Duration::from_secs_f64(0.3)) {
                    if SystemTime::now().duration_since(t).unwrap() < Duration::from_secs_f64(0.2) {
                        screen_print!(sec: 0.2, col: Color::SEA_GREEN, "board trigger click");
                        //println!("smart trigger");
                        sim.state.plate_click(t, *pos, 1.0);
                        sim.plate = sim.state.plate_state;
                        break;
                    }
                }
            }
            // for (i, tp) in trigger.plate_points.iter_mut().enumerate() {
            //     if let Some(t) = peeper.0.lock().unwrap().flush_or_add(&tp.points, Color::rgb(settings.plate_color[0], settings.plate_color[1], settings.plate_color[2]), settings.plate_threshold) {
            //         // let cvec = PLATE_COL;

            //         // let tvec = Vec3::new(col.r(), col.g(), col.b());

            //         // let diff = cvec - tvec;
            //         let time_since_last = tp.prev.map(|x| t.duration_since(x).unwrap().as_secs_f64());

            //         if time_since_last.unwrap_or(MIN_INTERVAL_PLATE) >= MIN_INTERVAL_PLATE {
            //             // Trigger.
            //             screen_print!(sec: 0.5, "plate trigger click, i={}", i);
            //             sim.state.plate_click(t, tp.pos, 1.0);
            //             sim.plate = sim.state.plate_state;
            //             tp.prev = Some(t);
            //         }
            //     }
            // }

            trigger.state = TriggerState::Active;
        }
        TriggerState::Setup(n) => {
            let ld = settings.dynamic_points.len();
            if n < ld {
                screen_print!(col: Color::YELLOW, "trigger setup - dynamic point, i={}, pos={}", n, settings.dynamic_points[n]);
                if let Some(click) = click {
                    trigger.dynamic_points.push(TriggerPoint {
                        points: vec![Point::new(click.x as f64, click.y as f64)],
                        pos: settings.dynamic_points[n as usize],
                        prev: None,
                    });
                    trigger.state = TriggerState::Setup(n + 1);
                }
            } else if n < ld + settings.plate_points.len() {
                let i = (n as usize - ld);
                screen_print!(col: Color::YELLOW, "trigger setup - plate point, i={}, pos={}", i, settings.plate_points[i]);
                if let Some(click) = click {
                    trigger.plate_points.push(TriggerPoint {
                        points: vec![Point::new(click.x as f64, click.y as f64)],
                        pos: settings.plate_points[i],
                        prev: None,
                    });
                    trigger.state = TriggerState::Setup(n + 1);
                }
            } else {
                screen_print!(col: Color::YELLOW, "trigger setup complete");
                trigger.state = TriggerState::Idle;
            }
        }
        TriggerState::PatternSetup => {
            if let Some(click) = click {
                if settings.dynamic_pattern.len() == settings.dynamic_points.len()
                    && settings.plate_pattern.len() == settings.plate_points.len()
                {
                    for (dp, pos) in settings
                        .dynamic_pattern
                        .iter()
                        .zip(settings.dynamic_points.iter())
                    {
                        trigger.dynamic_points.push(TriggerPoint {
                            points: vec![Point::new(
                                click.x as f64 + dp[0] as f64,
                                click.y as f64 + dp[1] as f64,
                            )],
                            pos: *pos,
                            prev: None,
                        });
                    }
                    for (tp, pos) in settings
                        .plate_pattern
                        .iter()
                        .zip(settings.plate_points.iter())
                    {
                        trigger.plate_points.push(TriggerPoint {
                            points: vec![Point::new(
                                click.x as f64 + tp[0] as f64,
                                click.y as f64 + tp[1] as f64,
                            )],
                            pos: *pos,
                            prev: None,
                        });
                    }
                    screen_print!(col: Color::YELLOW, "trigger setup using pattern");
                    trigger.state = TriggerState::Idle;
                } else {
                    screen_print!(col: Color::YELLOW, "trigger setup failed, number of points in pattern must match");
                    trigger.state = TriggerState::Idle;
                }
            }
        }
        _ => {}
    }
}

pub fn trigger_pos(n: u32) -> f64 {
    std::f64::consts::FRAC_PI_2 * n as f64
}

#[derive(Event)]
pub struct GlobalClickEvent {
    pos: Vec2,
    time: SystemTime,
}

#[derive(Clone)]
pub struct PixelBuffer {
    target: Color,
    time: Option<SystemTime>,
    points: Vec<autopilot::geometry::Point>,
    threshold: f32,
}

#[derive(Resource)]
pub struct PeeperRes(Arc<Mutex<PixelPeeper>>);
pub struct PixelPeeper {
    buffers: Vec<PixelBuffer>,
}

impl PixelPeeper {
    pub fn update(peeper: &Arc<Mutex<Self>>) {
        let buffers: Vec<PixelBuffer> = { peeper.lock().unwrap().buffers.clone() };
        'o: for b in buffers {
            for point in b.points.iter() {
                if let Ok(col) = autopilot::screen::get_color(*point) {
                    let nowc = Color::rgb_u8(col[0], col[1], col[2]);
                    let cvec = Vec3::new(nowc.r(), nowc.g(), nowc.b());
                    let tvec = Vec3::new(b.target.r(), b.target.g(), b.target.b());

                    if tvec.distance(cvec) < b.threshold {
                        peeper.lock().unwrap().activate(&b.points);
                        continue 'o;
                    }
                }
            }
        }
    }

    pub fn activate(&mut self, points: &[Point]) {
        for b in self.buffers.iter_mut() {
            if &b.points == points {
                if b.time.is_none() {
                    b.time = Some(SystemTime::now());
                }
            }
        }
    }

    pub fn flush_pixel(&mut self, points: &[Point]) -> Option<SystemTime> {
        for buf in self.buffers.iter_mut() {
            if &buf.points == points {
                return buf.time.take();
            }
        }
        None
    }

    pub fn flush_or_add(
        &mut self,
        points: &[Point],
        target: Color,
        threshold: f32,
    ) -> Option<SystemTime> {
        for buf in self.buffers.iter_mut() {
            if buf.points == points {
                return buf.time.take();
            }
        }
        self.add_pixel(points, target, threshold);
        None
    }

    pub fn add_pixel(&mut self, points: &[Point], target: Color, threshold: f32) {
        for buf in self.buffers.iter_mut() {
            if &buf.points == points {
                return;
            }
        }
        self.buffers.push(PixelBuffer {
            target,
            points: points.to_owned(),
            time: None,
            threshold,
        });
    }

    pub fn remove_pixel(&mut self, points: &[Point]) {
        self.buffers.retain(|x| &x.points == points);
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    pub fn peeper_daemon(peeper: Arc<Mutex<Self>>) {
        loop {
            PixelPeeper::update(&peeper);
            //std::thread::sleep(Duration::from_nanos(100));
        }
    }
}

pub fn average_color(rect: [i32; 4]) -> Option<Color> {
    let portion = autopilot::bitmap::capture_screen_portion(autopilot::geometry::Rect {
        origin: autopilot::geometry::Point::new(rect[0] as f64, rect[1] as f64),
        size: autopilot::geometry::Size::new(rect[2] as f64, rect[3] as f64),
    })
    .ok()?;

    let mut total_r: u64 = 0;
    let mut total_g: u64 = 0;
    let mut total_b: u64 = 0;

    let mut count = 0;
    for pixel in portion.image.as_rgb8()?.pixels().into_iter() {
        total_r += pixel.0[0] as u64;
        total_g += pixel.0[1] as u64;
        total_b += pixel.0[2] as u64;
        count += 1;
    }

    total_r /= count;
    total_g /= count;
    total_b /= count;

    Some(Color::rgb_u8(total_r as u8, total_g as u8, total_b as u8))
}

pub fn beep() {
    std::thread::spawn(|| {
        println!("beep");
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let sink = rodio::Sink::try_new(&stream_handle).unwrap();

        // Add a dummy source of the sake of the example.
        let source = SineWave::new(440.0)
            .take_duration(Duration::from_secs_f32(0.25))
            .amplify(0.20);
        sink.append(source);
        sink.sleep_until_end();
    });
}

impl HistoryRes {
    /// Assuming we bet a constant amount each time, what would be the returns given this history.
    /// Returns (actual, expected).
    pub fn constant_bet_returns(&self) -> (f64, f64) {
        let mut returns = 0.0;
        let mut expected = 0.0;

        for rec in self.0.iter() {
            if let Some(bet) = &rec.bet {
                if bet.0.contains(&rec.outcome.n) {
                    returns += (36.0 / (bet.0.len() as f64)) - 1.0;
                } else {
                    returns -= 1.0;
                }
                expected -= 1.0 - 36.0 / 37.0;
            }
        }

        (returns, expected)
    }

    pub fn winrate(&self) -> (f64, f64) {
        let mut wins = 0.0;
        let mut expected = 0.0;

        let mut count = 0;

        for rec in self.0.iter() {
            if let Some(bet) = &rec.bet {
                count += 1;
                if bet.0.contains(&rec.outcome.n) {
                    wins += 1.0;
                }
                expected += bet.0.len() as f64 / 37.0;
            }
        }

        (wins / count as f64, expected / count as f64)
    }
}

pub fn heartbeat_interaction(point: [i32; 2], undo_coords: [i32; 2]) {
    std::thread::spawn(move || {
        println!("Heartbeat");
        // Create an instance of Enigo
        let mut enigo = enigo::Enigo::new(&Default::default()).unwrap();

        // Move the mouse to the position (x=500, y=500)
        enigo.move_mouse(point[0], point[1], enigo::Coordinate::Abs);
        std::thread::sleep(Duration::from_millis(5));

        // Simulate a left mouse button click
        enigo.button(enigo::Button::Left, enigo::Direction::Click);

        std::thread::sleep(Duration::from_millis(500));

        enigo.move_mouse(undo_coords[0], undo_coords[1], enigo::Coordinate::Abs);
        std::thread::sleep(Duration::from_millis(5));

        enigo.button(enigo::Button::Left, enigo::Direction::Click);
    });
}
