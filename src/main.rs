// Plain egui frontend for DrawbackChess2 using Pleco
// v 0.2 -- 2024

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use eframe::egui;
use pleco::{Board, Player, PieceType, Rank, SQ, Piece}; // Standard types (Removed File)
use pleco::core::piece_move::BitMove; // Correct import for the move struct
use pleco::core::move_list::MoveList; // Import MoveList
use pleco::bots::JamboreeSearcher; // Bot implementation
use pleco::tools::Searcher; // Import the Searcher trait
use pleco::tools::eval::Eval; // Import Eval for static evaluation
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const ENGINE: u8 = 1;
const HUMAN: u8 = 0;

// Maps Pleco Piece to a display character index
fn piece_to_char_index(piece: Piece) -> usize {
    match piece {
        Piece::BlackKing => 0,
        Piece::BlackQueen => 1,
        Piece::BlackRook => 2,
        Piece::BlackBishop => 3,
        Piece::BlackKnight => 4,
        Piece::BlackPawn => 5,
        Piece::None => 6, // Empty square index
        Piece::WhitePawn => 7,
        Piece::WhiteKnight => 8,
        Piece::WhiteBishop => 9,
        Piece::WhiteRook => 10,
        Piece::WhiteQueen => 11,
        Piece::WhiteKing => 12
    }
}

const FIGURES: [&str; 13] = [
    "♚", "♛", "♜", "♝", "♞", "♟", // Black pieces (indices 0-5)
    "",                            // Empty square (index 6)
    "♙", "♘", "♗", "♖", "♕", "♔", // White pieces (indices 7-12)
];

// State machine for UI and game flow
const STATE_UZ: i32 = -2; // Determine whose turn it is
const STATE_UX: i32 = -1; // Game terminated (checkmate/stalemate/draw)
const STATE_U0: i32 = 0;  // Waiting for human player's first click (source square)
const STATE_U1: i32 = 1;  // Waiting for human player's second click (destination square)
const STATE_U2: i32 = 2;  // Engine's turn to think (start thread)
const STATE_U3: i32 = 3;  // Engine is thinking (waiting for thread result)

const BOOL_TO_ENGINE: [u8; 2] = [HUMAN, ENGINE];
const BOOL_TO_STATE: [i32; 2] = [STATE_U0, STATE_U2]; // Map player type (Human/Engine) to state

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "DrawbackChess2 (Pleco)",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::new(MyApp::default())
        }),
    )
}

struct MyApp {
    board: Arc<Mutex<Board>>, // Use Pleco's Board
    last_move_msg: String,
    last_score_msg: String,
    last_depth_msg: String,
    rotated: bool,
    time_per_move: f32,        // Seconds per move for the engine
    max_engine_depth: u16,     // Max search depth for the engine
    tagged: [u8; 64],          // Board highlighting (0=none, 1=possible dest, 2=last move)
    state: i32,                // Current UI/game state
    players: [u8; 2],          // 0 = Human, 1 = Engine (index 0=White, 1=Black)
    engine_plays_white: bool,  // UI checkbox state
    engine_plays_black: bool,  // UI checkbox state
    p0: Option<SQ>,            // Source square selected by human
    new_game: bool,            // Flag to reset the game
    rx: Option<mpsc::Receiver<(Option<BitMove>, u16)>>, // Receiver for (Option<move>, depth)
    move_count: u16,          // Track move number for progressive time management
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            board: Arc::new(Mutex::new(Board::start_pos())), // Start with standard position
            last_move_msg: "DrawbackChess2 (Pleco)".to_owned(),
            last_score_msg: "".to_owned(),
            last_depth_msg: "".to_owned(),
            time_per_move: 1.0,     // Default reduced to 1 second per move for more responsive gameplay
            max_engine_depth: 8,     // Default reduced to 8 for faster play
            rotated: false, // Default: White at bottom
            tagged: [0; 64],
            players: [ENGINE, ENGINE], // Default: AI vs AI
            p0: None,
            state: STATE_UZ,
            new_game: true, // Start a new game on launch
            engine_plays_white: true,
            engine_plays_black: true,
            rx: None,
            move_count: 0,
        }
    }
}

// Helper to convert Pleco SQ to 0-63 index
fn sq_to_idx(sq: SQ) -> usize {
    sq.file() as usize + sq.rank() as usize * 8
}

// Helper to convert 0-63 index to Pleco SQ
fn idx_to_sq(idx: usize) -> SQ {
    SQ(idx as u8) // Directly construct SQ from index
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(1.5);

        // --- Game Reset Logic ---
        if self.new_game {
            if let Ok(mut board) = self.board.try_lock() {
                *board = Board::start_pos();
                self.new_game = false;
                self.state = STATE_UZ;
                self.tagged = [0; 64];
                self.last_move_msg = "New Game Started".to_owned();
                self.last_score_msg = "".to_owned();
                self.last_depth_msg = "".to_owned();
                self.p0 = None;
                self.move_count = 0;
                
                // Clear any existing engine search
                self.rx = None;
            } else {
                ctx.request_repaint();
                return;
            }
        }

        // --- UI Drawing ---
        let mut clicked_sq: Option<SQ> = None;

        egui::SidePanel::left("side_panel")
            .min_width(200.0)
            .show(ctx, |ui| {
                ui.heading("DrawbackChess2 (Pleco)");
                ui.separator();
                ui.label(&self.last_move_msg);
                if !self.last_score_msg.is_empty() {
                    ui.label(&self.last_score_msg).on_hover_text(
                        "Position evaluation in centipawns. Positive values favor White.",
                    );
                }
                if !self.last_depth_msg.is_empty() {
                    ui.label(&self.last_depth_msg).on_hover_text(
                        "Search depth reached by the engine.",
                    );
                }
                ui.separator();
                ui.label("Status:");
                match self.state {
                    STATE_UZ => ui.label("Determining next player..."),
                    STATE_UX => ui.label("Game Over."),
                    STATE_U0 => ui.label("Waiting for source square..."),
                    STATE_U1 => ui.label("Waiting for destination square..."),
                    STATE_U2 => ui.label("Starting engine search..."),
                    STATE_U3 => ui.label("Engine thinking..."),
                    _ => ui.label("Unknown state"),
                };
                ui.separator();
                ui.add(egui::Slider::new(&mut self.time_per_move, 0.1..=5.0).text("Seconds per move"))
                    .on_hover_text("Engine will use exactly this much time for each move.");
                
                ui.label(format!("Move: {}", self.move_count));
                
                if ui.button("Rotate Board").clicked() {
                    self.rotated ^= true;
                }
                if ui.button("New Game").clicked() {
                    self.new_game = true;
                }
                if ui.checkbox(&mut self.engine_plays_white, "Engine plays White").changed() {
                    self.players[0] = BOOL_TO_ENGINE[self.engine_plays_white as usize];
                    if self.state != STATE_U2 && self.state != STATE_U3 {
                        self.state = STATE_UZ;
                    }
                }
                if ui.checkbox(&mut self.engine_plays_black, "Engine plays Black").changed() {
                    self.players[1] = BOOL_TO_ENGINE[self.engine_plays_black as usize];
                     if self.state != STATE_U2 && self.state != STATE_U3 {
                        self.state = STATE_UZ;
                    }
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = ui.available_size();
            let board_size = available_size.min_elem();
            let square_size = board_size / 8.0;

            let (rect, _response) =
                ui.allocate_exact_size(egui::vec2(board_size, board_size), egui::Sense::hover());

            let painter = ui.painter_at(rect);

            if let Ok(board) = self.board.try_lock() {
                for idx in 0..64 {
                    let sq = idx_to_sq(idx);
                    let rank_idx = sq.rank() as usize; // Get 0-7 index by casting
                    let file_idx = sq.file() as usize; // Get 0-7 index by casting

                    // Determine screen coordinates based on rotation
                    let screen_rank = if self.rotated { 7 - rank_idx } else { rank_idx };
                    let screen_file = if self.rotated { 7 - file_idx } else { file_idx };

                    let top_left = rect.min + egui::vec2(screen_file as f32 * square_size, screen_rank as f32 * square_size);
                    let square_rect = egui::Rect::from_min_size(top_left, egui::vec2(square_size, square_size));

                    // --- Square Color and Highlighting ---
                    let highlight_level = self.tagged[idx];
                    let base_color = if (rank_idx + file_idx) % 2 == 0 {
                        egui::Color32::WHITE // Light squares
                    } else {
                        egui::Color32::from_rgb(180, 180, 180) // Dark squares
                    };
                    let square_color = match highlight_level {
                        1 => egui::Color32::LIGHT_GREEN, // Possible move
                        2 => egui::Color32::LIGHT_YELLOW, // Last move
                        _ => base_color,
                    };

                    painter.rect_filled(square_rect, 0.0, square_color);

                     // --- Draw Piece ---
                     let piece = board.piece_at_sq(sq);
                     if piece != Piece::None {
                         let char_index = piece_to_char_index(piece);
                         painter.text(
                             square_rect.center(),
                             egui::Align2::CENTER_CENTER,
                             FIGURES[char_index],
                             egui::FontId::proportional(square_size * 0.8),
                             egui::Color32::BLACK,
                         );
                     }

                    // --- Handle Clicks ---
                    let response = ui.allocate_rect(square_rect, egui::Sense::click());
                    if response.clicked() {
                        clicked_sq = Some(sq);
                    }
                }
            } else {
                 ui.label("Waiting for board lock...");
            }

            // Request repaint if engine is thinking
             if self.state == STATE_U3 {
                 ui.ctx().request_repaint_after(Duration::from_millis(100));
             }
        });

        // --- State Machine Logic ---
        self.handle_state_machine(ctx, clicked_sq);
    }
}

impl MyApp {
    fn handle_state_machine(&mut self, ctx: &egui::Context, clicked_sq: Option<SQ>) {
        match self.state {
            STATE_UX => {
                // Game terminated, do nothing until new game
            },
            STATE_UZ => {
                // Determine next player and state
                if let Ok(board) = self.board.try_lock() {
                     let current_player_index = board.turn() as usize; // 0 for White, 1 for Black
                     self.state = BOOL_TO_STATE[self.players[current_player_index] as usize];
                } else {
                     ctx.request_repaint(); // Try again next frame if lock failed
                }
            },
            STATE_U0 => {
                // Waiting for Human Source Click
                self.handle_source_click(ctx, clicked_sq);
            },
            STATE_U1 => {
                // Waiting for Human Destination Click
                self.handle_destination_click(ctx, clicked_sq);
            },
            STATE_U2 => {
                // Start Engine Search
                self.start_engine_search(ctx);
            },
            STATE_U3 => {
                // Engine Thinking
                self.handle_engine_thinking(ctx);
            },
            _ => self.state = STATE_UZ, // Invalid state, reset
        }
    }
    
    fn handle_source_click(&mut self, ctx: &egui::Context, clicked_sq: Option<SQ>) {
        if let Some(sq) = clicked_sq {
            if let Ok(board) = self.board.try_lock() {
                let piece_on_sq = board.piece_at_sq(sq);
                // Check if clicked square has a piece of the current player
                if piece_on_sq != Piece::None && piece_on_sq.player().map_or(false, |p| p == board.turn()) {
                    self.p0 = Some(sq);
                    self.tagged = [0; 64]; // Clear old highlights
                    self.tagged[sq_to_idx(sq)] = 2; // Highlight source square

                    // Highlight legal destinations
                    let moves: MoveList = board.generate_moves();
                    for m in moves.iter() {
                        if (*m).get_src() == sq {
                            self.tagged[sq_to_idx((*m).get_dest())] = 1;
                        }
                    }
                    self.state = STATE_U1; // Move to waiting for destination
                }
            } else {
                ctx.request_repaint();
            }
        }
    }
    
    fn handle_destination_click(&mut self, ctx: &egui::Context, clicked_sq: Option<SQ>) {
        if let Some(dest_sq) = clicked_sq {
            if let Some(src_sq) = self.p0 {
                // Attempt to find the move
                let mut move_to_make: Option<BitMove> = None;
                if let Ok(board) = self.board.try_lock() {
                    let legal_moves: MoveList = board.generate_moves();
                    // Check for promotion first
                    let piece_on_src = board.piece_at_sq(src_sq);
                    let is_pawn = piece_on_src.type_of() == PieceType::P;
                    let promotion_rank = if board.turn() == Player::White { Rank::R8 } else { Rank::R1 };
                    let is_promotion = is_pawn && dest_sq.rank() == promotion_rank;

                    for m in legal_moves.iter() {
                        if (*m).get_src() == src_sq && (*m).get_dest() == dest_sq {
                            // Basic move match
                            if is_promotion {
                                // Found a potential promotion, check if it's queen promo
                                if (*m).is_promo() && (*m).promo_piece() == PieceType::Q {
                                    move_to_make = Some(*m);
                                    break;
                                }
                            } else {
                                // Non-promotion move matches
                                move_to_make = Some(*m);
                                break;
                            }
                        }
                    }

                    // If no queen promotion found, but it *should* be a promotion, create queen promo
                    if move_to_make.is_none() && is_promotion {
                        let target_piece = board.piece_at_sq(dest_sq);
                        let promo_flag = if target_piece != Piece::None {
                            BitMove::FLAG_PROMO_CAP_Q // Capture promotion
                        } else {
                            BitMove::FLAG_PROMO_Q // Simple promotion
                        };
                        move_to_make = Some(BitMove::make(promo_flag, src_sq, dest_sq));
                    }
                } else {
                    ctx.request_repaint();
                    return;
                }

                if let Some(move_to_apply) = move_to_make {
                    // Apply the move
                    if let Ok(mut board) = self.board.try_lock() {
                        let move_string = move_to_apply.stringify();
                        board.apply_move(move_to_apply);
                        self.move_count += 1; // Increment move counter

                        // Update UI
                        self.last_move_msg = format!("Human: {}", move_string);
                        self.last_score_msg = "".to_owned();
                        self.last_depth_msg = "".to_owned();
                        self.tagged = [0; 64];
                        self.tagged[sq_to_idx(move_to_apply.get_src())] = 2;
                        self.tagged[sq_to_idx(move_to_apply.get_dest())] = 2;
                        self.p0 = None;
                        self.state = STATE_UZ; // Go back to determine next player
                    } else {
                        ctx.request_repaint();
                    }
                } else {
                    // Invalid move click or clicked source square again - reset to state U0
                    self.p0 = None;
                    self.tagged = [0; 64];
                    self.state = STATE_UZ; // Reset highlights and state
                }
            } else {
                // Should not happen if p0 is None in state U1, but reset just in case
                self.p0 = None;
                self.tagged = [0; 64];
                self.state = STATE_UZ;
            }
        }
    }
    
    fn start_engine_search(&mut self, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel::<(Option<BitMove>, u16)>();
        self.rx = Some(rx);
        let time_ms = (self.time_per_move * 1000.0) as u64;

        self.state = STATE_U3; // Move to thinking state
        
        // Start time measurement to display in UI
        let start_time = Instant::now();
        
        // Update UI to show we're starting the search
        self.last_depth_msg = "Starting search...".to_owned();
        
        // Make a copy of the board BEFORE spawning thread to avoid locking issues
        let board_copy = if let Ok(board) = self.board.try_lock() {
            board.clone()
        } else {
            // Failed to get initial lock, will try again next frame
            self.state = STATE_U2;
            ctx.request_repaint();
            return;
        };

        thread::spawn(move || {
            // No need to lock the board here since we're using our copy
            let board_to_search = board_copy;
            
            // Use the time-based search directly
            let move_found = JamboreeSearcher::best_move_time(board_to_search, time_ms);
            
            // Send the result back with an estimated depth (we don't know the actual depth)
            let estimated_depth = 8; // Reasonable guess
            let _ = tx.send((Some(move_found), estimated_depth));
        });
    }
    
    fn handle_engine_thinking(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.rx {
            match rx.try_recv() {
                Ok((opt_move, depth)) => {
                    // Check if it's a final move or just a depth update
                    if let Some(applied_move) = opt_move {
                        // --- Apply Final Move ---
                        if let Ok(mut board) = self.board.try_lock() {
                            // Get move string *before* applying the move
                            let move_string = applied_move.stringify();
                            board.apply_move(applied_move);
                            self.move_count += 1; // Increment move counter

                            // Calculate static eval *after* move is applied
                            let static_eval = Eval::eval_low(&board) as i16;

                            // Update UI
                            self.last_move_msg = format!("Engine: {}", move_string);
                            self.last_score_msg = format!("Score: {:.2}", static_eval as f32 / 100.0);
                            self.last_depth_msg = format!("Depth: ~{}", depth);

                            self.tagged = [0; 64];
                            self.tagged[sq_to_idx(applied_move.get_src())] = 2;
                            self.tagged[sq_to_idx(applied_move.get_dest())] = 2;
                            self.p0 = None;

                            // Check for game over after engine move
                            if board.checkmate() {
                                self.last_move_msg.push_str(" Checkmate!");
                                self.state = STATE_UX;
                            } else if board.stalemate() || board.fifty_move_rule() {
                                self.last_move_msg.push_str(" Draw by 50-move or Stalemate!");
                                self.state = STATE_UX;
                            } else {
                                self.state = STATE_UZ; // Continue game
                            }

                            // Clear receiver only after processing final move
                            self.rx = None;
                        } else {
                            ctx.request_repaint(); // Try lock again next frame
                        }
                    } else {
                        // --- Depth Update ---
                        if depth > 0 {
                            self.last_depth_msg = format!("Depth: {} (searching...)", depth);
                            // Keep requesting repaints to check for more messages
                            ctx.request_repaint_after(Duration::from_millis(50));
                        } else {
                            // Depth 0 with None move signals failure from engine thread
                            self.last_move_msg = "Engine failed to find a move.".to_owned();
                            self.state = STATE_UX; // End game
                            self.rx = None;
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Engine still thinking, request repaint to check again soon
                    ctx.request_repaint_after(Duration::from_millis(50));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Thread panicked or sender dropped unexpectedly
                    self.last_move_msg = "Engine search thread error.".to_owned();
                    self.state = STATE_UX;
                    self.rx = None;
                }
            }
        } else {
            // Should not happen in state U3, reset state
            self.state = STATE_UZ;
        }
    }
}
