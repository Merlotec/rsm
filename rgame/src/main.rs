use autopilot::{bitmap, geometry::Point};
use bevy::{prelude::*, input::{keyboard::KeyboardInput, ButtonState}};
use motion::SmartTrigger;
use nalgebra::SVector;
use rodio::{source::SineWave, OutputStream, Source};
use spindle::{game::{DynamicState, PlateState}, sim::{SimParams, SimState}};
use std::{ops::Deref, process::exit, sync::{mpsc::{Receiver, Sender}, Mutex}, time::{Duration, SystemTime}};
use bevy_debug_text_overlay::{screen_print, OverlayPlugin};
use bevy::input::mouse::MouseButtonInput;
use std::sync::Arc;

//pub mod auto;
pub mod motion;

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
    dynamic_points: Vec<f64>,
    plate_points: Vec<f64>,
    dynamic_color: [f32; 3],
    dynamic_threshold: f32,
    plate_color: [f32; 3],
    plate_threshold: f32,
    dynamic_pattern: Vec<[f32; 2]>,
    plate_pattern: Vec<[f32; 2]>,
    board_rect_a: [i32; 4],
    board_rect_b: [i32; 4],
    board_a_point: f64,
    board_b_point: f64,
}

impl Settings {
    pub fn write_params(&mut self, path: &str, params: SimParams) {
        self.dynamic_acc = params.dynamic_acc;
        self.plate_acc = params.plate_acc;
        self.att = params.att;
        self.min_vel = params.min_vel;
        self.end_t = params.end_t;
        self.end_d = params.end_d;
        self.k = params.k;
        self.dynamic_weights = params.dynamic_weights.into();


        if let Ok(str) = serde_json::to_string(self) {
            let _ = std::fs::write(path, str);
        }
    }
}

fn default_settings() -> Settings {
     Settings {
        plate_acc: -0.00326, dynamic_acc: -0.006, min_vel: 2.65, k: -0.0073, att: 2.5, end_d: 1.0, end_t: 2.5, 
        dynamic_points: vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, std::f64::consts::FRAC_PI_2 * 3.0],
        plate_points: vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, std::f64::consts::FRAC_PI_2 * 3.0],
        dynamic_color: [0.9, 0.9, 0.9],
        dynamic_threshold: 0.2,
        plate_color: [0.2, 0.45, 0.2],
        plate_threshold: 0.2,
        dynamic_pattern: vec![[0.0, -270.0], [270.0, 0.0], [0.0, 270.0],[-270.0, 0.0]],
        plate_pattern: vec![[0.0, -150.0], [150.0, 0.0], [0.0, 150.0],[-150.0, 0.0]],
        dynamic_weights: Default::default(),
        dynamic_rect: [933, 235, 90, 50],
        board_rect_a: [1031, 442, 85, 23],
        board_rect_b: [1543, 432, 140, 97],
        board_a_point: 0.0,
        board_b_point: std::f64::consts::FRAC_PI_2,
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
    let settings = load_settings();

    let peeper = Arc::new(Mutex::new(PixelPeeper { buffers: Vec::new() }));
    let peeper_c = peeper.clone();
    std::thread::spawn(move || {
        PixelPeeper::peeper_daemon(peeper_c);
    });

    let sm = Arc::new(SmartTrigger::new(motion::MotionDetector::new(settings.dynamic_rect[0], settings.dynamic_rect[1], settings.dynamic_rect[2], settings.dynamic_rect[3])));
    let smart_trigger = TriggerRes(sm.clone());

    std::thread::spawn(move || {
        SmartTrigger::daemon(sm);
    });

    let smba = Arc::new(SmartTrigger::new(motion::MotionDetector::new_green(settings.board_rect_a[0], settings.board_rect_a[1], settings.board_rect_a[2], settings.board_rect_a[3])));


    let smbb = Arc::new(SmartTrigger::new(motion::MotionDetector::new_green(settings.board_rect_b[0], settings.board_rect_b[1], settings.board_rect_b[2], settings.board_rect_b[3])));
    
    let board_trigger = BoardTriggerRes(vec![(smba.clone(), settings.board_a_point),(smbb.clone(), settings.board_b_point)]);
    
    std::thread::spawn(move || {
        SmartTrigger::daemon(smba);
    });

    std::thread::spawn(move || {
        SmartTrigger::daemon(smbb);
    });


    App::new()
        .add_event::<GlobalClickEvent>()
        .add_plugins(DefaultPlugins)
        .add_plugins(OverlayPlugin { font_size: 18.0, ..default() })
        .add_systems(Startup, create_board)
        .add_systems(Update, (input, update_plate, update_dynamic, auto_trigger))
        .insert_resource(PeeperRes(peeper))
        .insert_resource(smart_trigger)
        .insert_resource(board_trigger)
        .insert_resource(settings)
        .run();

}

pub const LAYOUT: [i8; 37] = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26];

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
}


#[derive(Resource)]
pub enum AutopilotState {
    InTrigger,
    AwaitingResult,
    Standby,
    Offline,
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
    commands.spawn((Plate, SpriteBundle { 
        texture: asset_server.load("plate.png"), 
        ..default() 
    })); 
    commands.spawn((Dynamic(None), SpriteBundle { 
        texture: asset_server.load("dynamic.png"), 
        transform: Transform::from_xyz(0.0, R, 0.0).with_scale(Vec3::new(0.02, 0.02, 0.02)),
        ..default() 
    })); 
    commands.insert_resource(Sim { state: new_state(true, &settings), plate: None, dynamic: None, clockwise: true });
    commands.insert_resource(AutoTrigger { dynamic_points: Vec::new(), plate_points: Vec::new(), state: TriggerState::Idle, prev_down: false });

}

fn new_state(clockwise: bool, settings: &Settings) -> SimState {
    SimState::new(clockwise, SystemTime::now(), 0.01, SimParams { plate_acc: settings.plate_acc, dynamic_acc: settings.dynamic_acc, min_vel: settings.min_vel, k: settings.k, att: settings.att, dynamic_weights: SVector::from(settings.dynamic_weights), end_t: settings.end_t, end_d: settings.end_d} )
}

fn print_solve(state: &SimState) {
    let sln = state.solve();
    if let Some(sln) = sln {
        let slot = state.plate_state.unwrap().slot_at_local_pos(sln, LAYOUT.len());
        screen_print!(sec: 20.0, col: Color::CYAN, "current sln: {}, i: {}, n: {}", sln, slot, LAYOUT[slot]);
    }
}

fn input(mut commands: Commands, mut key_evr: EventReader<KeyboardInput>, mut sim: ResMut<Sim>, mut settings: ResMut<Settings>, mut trigger: ResMut<AutoTrigger>, mut peeper: ResMut<PeeperRes>, smart_trigger: Res<TriggerRes>, board_trigger: Res<BoardTriggerRes>, mut set: ParamSet<(Query<(&Plate, &mut Transform)>, Query<(&mut Dynamic, &mut Transform)>)>) {
    let mut reset: bool = false;
    for ev in key_evr.iter() {
        if let ButtonState::Pressed = ev.state {
            match ev.key_code {
                Some(KeyCode::P) => {
                    sim.state.plate_click(SystemTime::now(), CLICK_POS, 0.3);
                    sim.plate = sim.state.plate_state;

                    print_solve(&sim.state);
                },
                Some(KeyCode::D) => {
                    sim.state.dynamic_click(SystemTime::now(), CLICK_POS, 0.3);
                    sim.dynamic = sim.state.dynamic_state;

                    print_solve(&sim.state);
                }
                Some(KeyCode::C) => {
                    sim.clockwise = !sim.clockwise;
                    screen_print!(sec: 5.0, col: Color::ORANGE, "clockwise = {}", sim.clockwise);
                    if sim.plate.is_none() && sim.dynamic.is_none() {
                        reset = true;
                    }
                },
                Some(KeyCode::T) => {
                    match trigger.state {
                        TriggerState::Idle => {
                            trigger.state = TriggerState::Setup(0);
                            trigger.reset();
                            screen_print!(sec: 5.0, col: Color::YELLOW, "TRIGGER SETUP");
                        },
                        TriggerState::Setup(_) => {
                            trigger.state = TriggerState::Idle;
                            screen_print!(sec: 5.0, col: Color::YELLOW_GREEN, "TRIGGER IDLE");
                        },
                        _ => {},
                    }
                },
                Some(KeyCode::Y) => {
                    match trigger.state {
                        TriggerState::Idle => {
                            trigger.state = TriggerState::PatternSetup;
                            trigger.reset();
                            screen_print!(sec: 5.0, col: Color::YELLOW, "TRIGGER PATTERN SETUP");
                        },
                        TriggerState::PatternSetup => {
                            trigger.state = TriggerState::Idle;
                            screen_print!(sec: 5.0, col: Color::YELLOW_GREEN, "TRIGGER IDLE");
                        },
                        _ => {},
                    }
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
                            for (tr, _) in board_trigger.0.iter() {
                                tr.clear();
                            } 
                            screen_print!(sec: 5.0, col: Color::RED, "TRIGGER ARMED");
                            
                        },
                        TriggerState::Armed => {
                            trigger.state = TriggerState::Idle;
                            screen_print!(sec: 5.0, col: Color::YELLOW_GREEN, "TRIGGER IDLE");
                        },
                        TriggerState::Setup(n) => {
                            trigger.reset();
                            if n >= settings.dynamic_points.len() + settings.plate_points.len() {
                                trigger.state = TriggerState::Armed;
                                screen_print!(sec: 5.0, col: Color::RED, "TRIGGER ARMED");
                            } else {
                                screen_print!(sec: 5.0, col: Color::YELLOW, "could not arm trigger - not enough points specified");

                            }
                        },
                        _ => {},
                    }
                },
                Some(KeyCode::S) => {
                    if let TriggerState::Active = trigger.state {
                        trigger.state = TriggerState::Idle;
                        screen_print!(sec: 5.0, col: Color::YELLOW_GREEN, "TRIGGER IDLE");
                        if let Some(out) = sim.state.solve() {
                            let slot = sim.state.plate_state.unwrap().slot_at_local_pos(out, LAYOUT.len());
    
                            screen_print!(sec: 20.0, col: Color::CYAN, "sln: {}, i: {}, n: {}", out, slot, LAYOUT[slot]);
                        }
                    }
                    
                },
                Some(KeyCode::V) => {
                    if let TriggerState::Active = trigger.state {
                        if let Some(dy) = sim.dynamic {
                            screen_print!(sec: 5.0, col: Color::PINK, "VELOCITY: {}", dy.vel);
                        }
                    }
                    
                },
                Some(KeyCode::L) => {
                    // Train the model.
                    screen_print!(sec: 5.0, col: Color::BLUE, "COMMENCED LEARNING ({} clicks)", sim.state.dynamic_clicks.len());
                    let old_k = sim.state.params.k;
                    let old_att = sim.state.params.att;
                    let outcome = sim.state.train(0.5, 600, 0.4, 0.01, 0.05);
                    
                    if let Some(outcome) = outcome {
                        screen_print!(sec: 5.0, col: Color::BLUE, "LEARNING COMPLETE");
                        sim.state.plot(&sim.state.params, "before.png");
                        sim.state.params = outcome;
                        sim.state.plot(&sim.state.params, "after.png");
                        settings.write_params("settings.json", outcome);
                    } else {
                        screen_print!(sec: 5.0, col: Color::RED, "LEARNING FAILED");
                    }
                },
                Some(KeyCode::R) => {
                    reset = true;
                    
                }
                _ => {},
            }
        }
    }

    if reset {
        *settings = load_settings();

        commands.insert_resource(Sim { state: new_state(sim.clockwise, &settings), plate: None, dynamic: None, clockwise: sim.clockwise });
        sim.plate = None;
        sim.dynamic = None;

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
    }
}

pub fn update_plate(time: Res<Time>, mut sim: ResMut<Sim>, mut plates: Query<(&Plate, &mut Transform)>) {
    for (plate, mut transform) in plates.iter_mut() {
        let ps = sim.plate;
        if let Some(ps) = ps {
            let next = ps.approximate(time.delta_seconds() as f64, sim.state.step);

            transform.rotation = Quat::from_rotation_z(-next.dis as f32);
            sim.plate = Some(next);
            screen_print!("pdis: {}, pvel: {}", ps.dis, ps.vel);
        }
    }
}

pub fn autopilot_system(mut ap: ResMut<AutopilotState>) {
    match ap.deref() {
        AutopilotState::Standby => {
            // wait for dark rect
        },
        AutopilotState::InTrigger => {
            // do nothing
        },
        AutopilotState::AwaitingResult => {
            // check result image
        },
        AutopilotState::Offline => {},
    }
}

pub fn update_dynamic(time: Res<Time>, mut sim: ResMut<Sim>, mut dynamics: Query<(&mut Dynamic, &mut Transform)>) {
    for (mut dynamic, mut transform) in dynamics.iter_mut() {
        if let Some(f) = dynamic.0 {
            let p = sim.plate.unwrap().dis + f;
            transform.translation.x = R_1 * p.sin() as f32;
            transform.translation.y = R_1 * p.cos() as f32;
        } else {
            let ds = sim.dynamic;
            if let Some(ds) = ds {
                let next = ds.approximate(time.delta_seconds() as f64, sim.state.step, sim.state.params.dynamic_acc, sim.state.params.k, sim.state.params.att, sim.state.params.dynamic_weights);
                transform.translation.x = R * next.dis.sin() as f32;
                transform.translation.y = R * next.dis.cos() as f32;

                sim.dynamic = Some(next);

                if ds.vel.abs() < sim.state.params.min_vel {
                    //beep();
                    if let Some(out) = sim.state.finalize(ds) {
                        dynamic.0 = Some(out);
                        let slot = sim.state.plate_state.unwrap().slot_at_local_pos(out, LAYOUT.len());

                        screen_print!(sec: 20.0, col: Color::RED, "sln: {}, i: {}, n: {}", out, slot, LAYOUT[slot]);

                    }
                }
    
                screen_print!("ddis: {}, dvel: {}", ds.dis, ds.vel);
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

fn auto_trigger(mut trigger: ResMut<AutoTrigger>, mut sim: ResMut<Sim>, mut peeper: ResMut<PeeperRes>, mut smart_trigger: ResMut<TriggerRes>, mut board_trigger: ResMut<BoardTriggerRes>, settings: Res<Settings>) {
    use device_query::MouseState;
    use device_query::{DeviceEvents, DeviceState, DeviceQuery};

    let device_state = DeviceState::new();
    let mouse: MouseState = device_state.get_mouse();

    let click: Option<Vec2> = if (mouse.button_pressed[1] == true && !trigger.prev_down) {
        Some(Vec2::new(mouse.coords.0 as f32, mouse.coords.1 as f32))
    } else {
        None
    };

    trigger.prev_down = mouse.button_pressed[1];

    match trigger.state {
        
        TriggerState::Active | TriggerState::Armed => {
            if let Some(t) = smart_trigger.0.flush_detect(Duration::from_secs_f64(0.1)) {
                screen_print!(sec: 0.2, "smart trigger click");
                //println!("smart trigger");
                sim.state.dynamic_click(t, 0.0, 1.0);
                sim.dynamic = sim.state.dynamic_state;
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
                if let Some(t) = tr.flush_detect(Duration::from_secs_f64(0.5)) {
                    screen_print!(sec: 0.2, "smart trigger click");
                    //println!("smart trigger");
                    sim.state.plate_click(t, *pos, 1.0);
                    sim.plate = sim.state.plate_state;
                    break;
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
        },
        TriggerState::Setup(n) => {
            
            let ld = settings.dynamic_points.len();
            if n < ld {
                screen_print!(col: Color::YELLOW, "trigger setup - dynamic point, i={}, pos={}", n, settings.dynamic_points[n]);
                if let Some(click) = click {
                    trigger.dynamic_points.push(TriggerPoint { points: vec![Point::new(click.x as f64, click.y as f64)], pos: settings.dynamic_points[n as usize], prev: None });
                    trigger.state = TriggerState::Setup(n + 1);
                }       
            } else if n < ld + settings.plate_points.len() {
                let i = (n as usize - ld);
                screen_print!(col: Color::YELLOW, "trigger setup - plate point, i={}, pos={}", i, settings.plate_points[i]);
                if let Some(click) = click {
                    trigger.plate_points.push(TriggerPoint { points: vec![Point::new(click.x as f64, click.y as f64)], pos: settings.plate_points[i], prev: None });
                    trigger.state = TriggerState::Setup(n + 1);
                }
            } else {
                screen_print!(col: Color::YELLOW, "trigger setup complete");
                trigger.state = TriggerState::Idle;
            }
            
        },
        TriggerState::PatternSetup => {
            if let Some(click) = click {
                if settings.dynamic_pattern.len() == settings.dynamic_points.len() && settings.plate_pattern.len() == settings.plate_points.len() {

                    for (dp, pos) in settings.dynamic_pattern.iter().zip(settings.dynamic_points.iter()) {
                        trigger.dynamic_points.push(TriggerPoint { points: vec![Point::new(click.x as f64 + dp[0] as f64, click.y as f64 + dp[1] as f64)], pos: *pos, prev: None });
                    }
                    for (tp, pos) in settings.plate_pattern.iter().zip(settings.plate_points.iter()) {
                        trigger.plate_points.push(TriggerPoint { points: vec![Point::new(click.x as f64 + tp[0] as f64, click.y as f64 + tp[1] as f64)], pos: *pos, prev: None });
                    }
                    screen_print!(col: Color::YELLOW, "trigger setup using pattern");
                    trigger.state = TriggerState::Idle;
                } else {
                    screen_print!(col: Color::YELLOW, "trigger setup failed, number of points in pattern must match");
                    trigger.state = TriggerState::Idle;
                }
            }

            
        },
        _ => {},
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
        let buffers: Vec<PixelBuffer> = {
            peeper.lock().unwrap().buffers.clone()
        };
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

    pub fn flush_or_add(&mut self, points: &[Point], target: Color, threshold: f32) -> Option<SystemTime> {
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
        self.buffers.push(PixelBuffer { target, points: points.to_owned(), time: None, threshold });
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

pub fn beep() {
    std::thread::spawn(|| {
        println!("beep");
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let sink = rodio::Sink::try_new(&stream_handle).unwrap();
        
        // Add a dummy source of the sake of the example.
        let source = SineWave::new(440.0).take_duration(Duration::from_secs_f32(0.25)).amplify(0.20);
        sink.append(source);
        sink.sleep_until_end();
    });
}

// pub struct NumberViewer {
//     number: Option<u32>,
//     changed: bool,
//     rect: autopilot::geometry::Rect,
//     active: bool,
// }


// impl NumberViewer {
//     pub fn new(rect: Rect) -> Self {
//         Self {
//             number: None,
//             changed: false,
//             rect: autopilot::geometry::Rect { origin: autopilot::geometry::Point::new(rect.min.x as f64, rect.min.y as f64), size: autopilot::geometry::Size::new(rect.width() as f64, rect.height() as f64) },
//             active: true,
//         }
//     }

//     pub fn update(this: &Arc<Mutex<Self>>) {
//         if this.lock().unwrap().active {
//             let bmp = bitmap::capture_screen_portion(this.lock().unwrap().rect);
//             if let Ok(bmp) = bmp {
//                 let img = bmp.image;

//                 let tmp_img_path = "/tmp/tmp_img.png"; // Adjust path as needed
//                 img.save(tmp_img_path).expect("Failed to save temporary image");

//                 // Use Tesseract to perform OCR
//                 let mut tess = tesseract::Tesseract::new(None, Some("eng"))
//                     .expect("Failed to create Tesseract instance");

//                 if let Ok(mut tess) = tess.set_image(tmp_img_path) {
//                     let mut this = this.lock().unwrap();
//                     let old_number: Option<u32> = this.number;
//                     this.number = tess.get_text().ok().and_then(|x| x.parse::<u32>().ok());

//                     this.changed = old_number != this.number;
//                 }
//             }
//         }
//     }
    
//     pub fn daemon(this: Arc<Mutex<Self>>) {
//         loop {
//             Self::update(&this);
//             //std::thread::sleep(Duration::from_nanos(100));
//         }
//     }
// }