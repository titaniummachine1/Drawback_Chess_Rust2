// Plain egui frontend for DrawbackChess2
// v 0.1 -- 2024
// Based on tiny-chess by Dr. Stefan Salewski (https://github.com/StefanSalewski/tiny-chess)
// Original code under MIT license

// this is a version with threading, using spawn and channels

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use eframe::egui;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
//use std::time::Duration;

mod engine;

const ENGINE: u8 = 1;
const HUMAN: u8 = 0;

const FIGURES: [&str; 13] = [
    "♚", "♛", "♜", "♝", "♞", "♟", "", "♙", "♘", "♗", "♖", "♕", "♔",
];

const STATE_UZ: i32 = -2; // state when engine or human player have made their move, so it's other sides turn
const STATE_UX: i32 = -1; // stable state, current game is terminated
const STATE_U0: i32 = 0;
const STATE_U1: i32 = 1;
const STATE_U2: i32 = 2;
const STATE_U3: i32 = 3;

const BOOL_TO_ENGINE: [u8; 2] = [HUMAN, ENGINE];
const BOOL_TO_STATE: [i32; 2] = [STATE_U0, STATE_U2];

fn _print_variable_type<K>(_: &K) {
    println!("{}", std::any::type_name::<K>())
}

fn _rot_180(b: engine::Board) -> engine::Board {
    let mut result: engine::Board = [0; 64];
    for (i, f) in b.iter().enumerate() {
        result[63 - i] = *f;
    }
    result
}

fn main() -> Result<(), eframe::Error> {
    //env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "DrawbackChess2",
        options,
        Box::new(|cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);
            // The closure should return Box<dyn eframe::App>
            Box::new(MyApp::default())
        }),
    )
}

struct MyApp {
    game: Arc<Mutex<engine::Game>>,
    last_move_msg: String,
    last_score_msg: String,
    last_depth_msg: String,
    rotated: bool,
    time_per_move: f32,
    tagged: engine::Board,
    state: engine::State,
    players: [u8; 2],
    engine_plays_white: bool,
    engine_plays_black: bool,
    p0: i32,
    new_game: bool,
    bbb: engine::Board,
    rx: Option<mpsc::Receiver<(engine::Move, i32)>>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            game: Arc::new(Mutex::new(engine::new_game())),
            last_move_msg: "DrawbackChess2".to_owned(),
            last_score_msg: "".to_owned(),
            last_depth_msg: "".to_owned(),
            time_per_move: 3.0,
            rotated: true,
            tagged: [0; 64],
            players: [ENGINE, ENGINE],
            p0: -1,
            state: STATE_UZ,
            bbb: [0; 64],
            new_game: true,
            engine_plays_white: true,
            engine_plays_black: true,
            rx: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(1.5);
        if let Ok(ref mut mutex) = self.game.try_lock() {
            if self.new_game {
                engine::reset_game(mutex);
                self.new_game = false;
                self.state = STATE_UZ;
                self.tagged = [0; 64];
                self.last_move_msg = "New Game Started".to_owned();
                self.last_score_msg = "".to_owned();
                self.last_depth_msg = "".to_owned();
            }
            self.bbb = engine::get_board(mutex);
            mutex.secs_per_move = self.time_per_move;
        }

        let mut x: i8 = -1;
        let mut y: i8 = -1;
        egui::SidePanel::left("side_panel")
            .min_width(200.0)
            .show(ctx, |ui| {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Title(
                    "DrawbackChess2".to_owned(),
                ));
                ui.heading(&self.last_move_msg);
                if !self.last_score_msg.is_empty() {
                    ui.label(&self.last_score_msg).on_hover_text("Position evaluation in centipawns. Positive values favor White, negative values favor Black.");
                }
                if !self.last_depth_msg.is_empty() {
                    ui.label(&self.last_depth_msg).on_hover_text("Search depth reached by the engine for the last evaluation.");
                }
                ui.separator();
                ui.add(egui::Slider::new(&mut self.time_per_move, 0.1..=5.0).text("Sec/move"));
                if ui.button("Rotate").clicked() {
                    self.rotated ^= true;
                    self.tagged.reverse();
                }
                if ui.button("Print movelist").clicked() {
                    engine::print_move_list(&self.game.lock().unwrap());
                }
                if ui.button("New Game").clicked() {
                    self.new_game = true;
                }
                if ui
                    .checkbox(&mut self.engine_plays_white, "Engine plays white")
                    .changed()
                {
                    self.players[0] = BOOL_TO_ENGINE[self.engine_plays_white as usize];
                    self.state = STATE_UZ;
                }
                if ui
                    .checkbox(&mut self.engine_plays_black, "Engine plays black")
                    .changed()
                {
                    self.players[1] = BOOL_TO_ENGINE[self.engine_plays_black as usize];
                    self.state = STATE_UZ;
                }
                ui.image(egui::include_image!("ferris.png"));
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.state == STATE_U2 {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Title(
                    " ... one moment please, reply is:".to_owned(),
                ));
            }
            let available_size = ui.available_size();
            let central_panel_rect = ui.min_rect();
            let center_x = central_panel_rect.center().x;
            let center_y = central_panel_rect.center().y;
            let mut responses = Vec::new();
            let board_size = available_size.min_elem();
            let square_size = board_size / 8.0;
            let board_top_left = egui::Pos2 {
                x: center_x - (4.0 * square_size),
                y: center_y - (4.0 * square_size),
            };
            for row in 0..8 {
                for col in 0..8 {
                    let p = col + row * 8;
                    let t = &self.tagged[p];
                    let h: u8;
                    if *t == 2 {
                        h = 25;
                    } else if *t == 1 {
                        h = 50;
                    } else {
                        h = 0;
                    }
                    let color = if (row + col) % 2 == 0 {
                        egui::Color32::from_rgb(255, 255, 255 - h)
                    } else {
                        egui::Color32::from_rgb(205, 205, 205 - h)
                    };
                    let top_left = egui::Pos2 {
                        x: board_top_left.x + (col as f32 * square_size),
                        y: board_top_left.y + (row as f32 * square_size),
                    };
                    let bottom_right = egui::Pos2 {
                        x: top_left.x + square_size,
                        y: top_left.y + square_size,
                    };
                    let rect = egui::Rect::from_two_pos(top_left, bottom_right);
                    let response = ui.allocate_rect(rect, egui::Sense::click());
                    let (r, c) = if self.rotated {
                        (7 - row, 7 - col)
                    } else {
                        (row, col)
                    };
                    responses.push((response, rect, color, c, r));
                }
            }
            let painter = ui.painter();
            for (response, rect, color, col, row) in responses {
                if response.clicked() {
                    x = col as i8;
                    y = row as i8;
                }
                painter.rect_filled(rect, 0.0, color);
                let text_pos = rect.center();
                let piece = FIGURES[(self.bbb[col + row * 8] + 6) as usize];
                painter.text(
                    text_pos,
                    egui::Align2::CENTER_CENTER,
                    piece,
                    egui::FontId::proportional(square_size * 0.9),
                    egui::Color32::BLACK,
                );
            }
            if self.state == STATE_U3 {
                ui.ctx().request_repaint();
            }
        });

        if self.state == STATE_UX {
            // game terminated
        } else if self.state == STATE_UZ {
            let next = self.game.lock().unwrap().move_counter as usize % 2;
            self.state = BOOL_TO_STATE[self.players[next] as usize];
        } else if self.state == STATE_U0 && x >= 0 {
            self.p0 = (x + y * 8) as i32;
            let h = self.p0 as i64;
            self.tagged = [0; 64];
            for i in engine::tag(&mut self.game.lock().unwrap(), h) {
                self.tagged[i.di as usize] = 1;
            }
            self.tagged[h as usize] = -1;
            if self.rotated {
                self.tagged.reverse();
            }
            self.state = STATE_U1;
        } else if self.state == STATE_U1 && x >= 0 {
            let p1 = x + y * 8;
            let h = self.p0;
            if h == p1 as i32
                || !engine::move_is_valid2(&mut self.game.lock().unwrap(), h as i64, p1 as i64)
            {
                self.last_move_msg = "Invalid move, ignored.".to_owned();
                self.last_score_msg = "".to_owned();
                self.last_depth_msg = "".to_owned();
                self.tagged = [0; 64];
                self.state = STATE_UZ;
                return;
            }
            let flag = engine::do_move(&mut self.game.lock().unwrap(), h as i8, p1 as i8, false);
            self.tagged = [0; 64];
            self.tagged[h as usize] = 2;
            self.tagged[p1 as usize] = 2;
            if self.rotated {
                self.tagged.reverse();
            }
            self.last_move_msg = engine::move_to_str(&mut self.game.lock().unwrap(), h as i8, p1 as i8, flag);
            self.last_score_msg = "".to_owned();
            self.last_depth_msg = "".to_owned();
            self.state = STATE_UZ;
        } else if self.state == STATE_U2 {
            self.state = STATE_U3;
            let (tx, rx) = mpsc::channel::<(engine::Move, i32)>();
            self.rx = Some(rx);
            let game_clone = self.game.clone();
            thread::spawn(move || {
                let (m, final_depth) = engine::reply(&mut game_clone.lock().unwrap());
                tx.send((m, final_depth as i32)).unwrap();
            });
        } else if self.state == STATE_U3 {
            if let Some(rx) = &self.rx {
                if let Ok((m, final_depth)) = rx.try_recv() {
                    self.tagged = [0; 64];
                    self.tagged[m.src as usize] = 2;
                    self.tagged[m.dst as usize] = 2;
                    if self.rotated {
                        self.tagged.reverse();
                    }
                    let flag = engine::do_move(
                        &mut self.game.lock().unwrap(),
                        m.src as i8,
                        m.dst as i8,
                        false,
                    );
                    self.last_move_msg = engine::move_to_str(
                        &mut self.game.lock().unwrap(),
                        m.src as i8,
                        m.dst as i8,
                        flag,
                    );
                    self.last_score_msg = format!("Score: {:.2}", m.score as f32 / 100.0);
                    self.last_depth_msg = format!("Depth: {}", final_depth);
                    
                    if m.score == engine::KING_VALUE as i64 {
                        self.last_move_msg.push_str("# Checkmate!");
                        self.state = STATE_UX;
                        return;
                    } else if m.score > engine::KING_VALUE_DIV_2 as i64 {
                        self.last_depth_msg += &format!(" (Mate in {})", (engine::KING_VALUE as i64 - m.score) / 2);
                    }
                    self.state = STATE_UZ;
                    self.rx = None;
                }
            }
        }
    }
}
// 312 lines
