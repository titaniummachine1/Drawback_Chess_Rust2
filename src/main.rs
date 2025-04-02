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
const STATE_U0: i32 = 0; // Waiting for human player's first click (source square)
const STATE_U1: i32 = 1; // Waiting for human player's second click (destination square)
const STATE_U2: i32 = 2; // Engine's turn to think (start thread)
const STATE_U3: i32 = 3; // Engine is thinking (waiting for thread result)

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
    tagged: [u8; 64],          // Board highlighting (0=none, 1=possible dest, 2=last move)
    state: i32,                // Current UI/game state
    players: [u8; 2],          // 0 = Human, 1 = Engine (index 0=White, 1=Black)
    engine_plays_white: bool,  // UI checkbox state
    engine_plays_black: bool,  // UI checkbox state
    p0: Option<SQ>,            // Source square selected by human
    new_game: bool,            // Flag to reset the game
    rx: Option<mpsc::Receiver<(Option<BitMove>, u16)>>, // Receiver for (Option<move>, depth)
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            board: Arc::new(Mutex::new(Board::start_pos())), // Start with standard position
            last_move_msg: "DrawbackChess2 (Pleco)".to_owned(),
            last_score_msg: "".to_owned(),
            last_depth_msg: "".to_owned(),
            time_per_move: 3.0, // Default time per move
            rotated: false, // Default: White at bottom
            tagged: [0; 64],
            players: [ENGINE, ENGINE], // Default: AI vs AI
            p0: None,
            state: STATE_UZ,
            new_game: true, // Start a new game on launch
            engine_plays_white: true,
            engine_plays_black: true,
            rx: None,
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
                // Ensure receiver is cleared if a game was aborted mid-search
                if let Some(rx) = self.rx.take() {
                    drop(rx); // Close the channel
                }
            } else {
                // If lock fails, request repaint and try again next frame
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
                ui.add(egui::Slider::new(&mut self.time_per_move, 0.1..=10.0).text("Sec/move"));
                if ui.button("Rotate Board").clicked() {
                    self.rotated ^= true;
                }
                if ui.button("New Game").clicked() {
                    self.new_game = true;
                }
                if ui.checkbox(&mut self.engine_plays_white, "Engine plays White").changed() {
                    self.players[0] = BOOL_TO_ENGINE[self.engine_plays_white as usize];
                    if self.state != STATE_U2 && self.state != STATE_U3 { // Don't interrupt engine
                        self.state = STATE_UZ;
                    }
                }
                if ui.checkbox(&mut self.engine_plays_black, "Engine plays Black").changed() {
                    self.players[1] = BOOL_TO_ENGINE[self.engine_plays_black as usize];
                     if self.state != STATE_U2 && self.state != STATE_U3 { // Don't interrupt engine
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
        //println!("Current state: {}", self.state);

        if self.state == STATE_UX {
            // Game terminated, do nothing until new game
        } else if self.state == STATE_UZ {
            println!("[State UZ] Determining next player...");
            // Determine next player and state
            if let Ok(board) = self.board.try_lock() {
                 let current_player_index = board.turn() as usize; // 0 for White, 1 for Black
                 self.state = BOOL_TO_STATE[self.players[current_player_index] as usize];
                 println!("  -> Next state: {}", self.state);
            } else {
                 ctx.request_repaint(); // Try again next frame if lock failed
            }
        } else if self.state == STATE_U0 { // Waiting for Human Source Click
            //println!("[State U0] Waiting for human source click...");
            if let Some(sq) = clicked_sq {
                 if let Ok(board) = self.board.try_lock() {
                    let piece_on_sq = board.piece_at_sq(sq);
                    // Check if clicked square has a piece of the current player
                    if piece_on_sq != Piece::None && piece_on_sq.player().map_or(false, |p| p == board.turn()) {
                         self.p0 = Some(sq);
                         self.tagged = [0; 64]; // Clear old highlights
                         self.tagged[sq_to_idx(sq)] = 2; // Highlight source square

                         // Highlight legal destinations
                         let moves: MoveList = board.generate_moves(); // Correct type
                         for m in moves.iter() { // Iterate over the MoveList
                             if (*m).get_src() == sq { // Dereference m
                                 self.tagged[sq_to_idx((*m).get_dest())] = 1; // Dereference m
                             }
                         }
                         println!("  -> Human selected source: {}, highlighting moves. Transitioning to U1", sq);
                         self.state = STATE_U1; // Move to waiting for destination
                    }
                 } else {
                      ctx.request_repaint();
                 }
            }
        } else if self.state == STATE_U1 { // Waiting for Human Destination Click
            if let Some(dest_sq) = clicked_sq {
                 if let Some(src_sq) = self.p0 {
                     // Attempt to find the move
                     let mut move_to_make: Option<BitMove> = None;
                     if let Ok(board) = self.board.try_lock() {
                         let legal_moves: MoveList = board.generate_moves(); // Correct type
                         // Check for promotion first (requires user input, default to Queen for now)
                         let piece_on_src = board.piece_at_sq(src_sq);
                         let is_pawn = piece_on_src.type_of() == PieceType::P;
                         let promotion_rank = if board.turn() == Player::White { Rank::R8 } else { Rank::R1 };
                         let is_promotion = is_pawn && dest_sq.rank() == promotion_rank;

                         for m in legal_moves.iter() { // Iterate over the MoveList
                             if (*m).get_src() == src_sq && (*m).get_dest() == dest_sq { // Dereference m
                                 // Basic move match
                                 if is_promotion {
                                     // Found a potential promotion, check if it's queen promo
                                     if (*m).is_promo() && (*m).promo_piece() == PieceType::Q { // Dereference m
                                         move_to_make = Some(*m); // Dereference m
                                         break;
                                     }
                                     // If not queen promo, keep searching (maybe user clicks again for knight?)
                                     // For now, we only handle queen promo automatically.
                                 } else {
                                     // Non-promotion move matches
                                     move_to_make = Some(*m); // Dereference m
                                     break;
                                 }
                             }
                         }

                         // If no queen promotion found, but it *should* be a promotion, create queen promo
                         if move_to_make.is_none() && is_promotion {
                              // Determine if it's a capture promotion
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
                     }


                     if let Some(move_to_apply) = move_to_make {
                          // Apply the move
                          if let Ok(mut board) = self.board.try_lock() {
                              let move_string = move_to_apply.stringify(); // Use stringify()
                              board.apply_move(move_to_apply);

                              // Update UI
                              self.last_move_msg = format!("Human: {}", move_string);
                              println!("[State U1] Human applied move: {}", move_string);
                              println!("  -> New Board FEN: {}", board.fen());
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
                         println!("[State U1] Invalid human move click. Resetting highlights.");
                         self.state = STATE_UZ; // Reset highlights and state
                     }
                 } else {
                     // Should not happen if p0 is None in state U1, but reset just in case
                     self.p0 = None;
                     self.tagged = [0; 64];
                     self.state = STATE_UZ;
                 }
            }
        } else if self.state == STATE_U2 { // Start Engine Search
            let (tx, rx) = mpsc::channel::<(Option<BitMove>, u16)>();
            self.rx = Some(rx);
            let board_clone = Arc::clone(&self.board); // Clone Arc for the thread
            let time_limit_millis = (self.time_per_move * 1000.0) as u128;

            self.state = STATE_U3; // Move to thinking state
            println!("[State U2] Starting engine thread. Transitioning to U3");

            thread::spawn(move || {
                // Use JamboreeSearcher (or replace with another Pleco bot if desired)
                // We need to find the best move within the time limit.
                // Pleco's `get_move` is suitable here.

                // Set the time limit for the search
                let start_time = Instant::now();
                let time_limit = Duration::from_millis(time_limit_millis as u64);

                // Use a loop with time checks for iterative deepening (conceptual)
                // Pleco's internal search likely handles this, but we manage the overall time limit.
                let mut best_move = BitMove::null();
                let mut best_move_depth: u16 = 0;
                let mut current_depth: u16 = 1;
                let max_search_depth: u16 = 7; // Further reduced max depth

                // Lock the Arc<Mutex<Board>> briefly to get a clone of the Board state.
                println!("  [Engine Thread] Locking board to clone...");
                // The lock is released immediately after the clone is created.
                let initial_board_state = board_clone.lock().unwrap().clone();
                println!("  [Engine Thread] Board cloned, lock released. Starting search loop...");

                while Instant::now().duration_since(start_time) < time_limit && current_depth <= max_search_depth {
                     // Send depth update *before* starting search for this depth
                     // Ignore send error if main thread closed receiver early
                     let _ = tx.send((None, current_depth));

                     // Clone the board state *from the initial clone* for this depth's search
                     // No need for the MutexGuard (board_lock) here anymore.
                     let board_to_search = initial_board_state.shallow_clone();

                     // Run the search for the current depth using the static method
                     // This blocks until the depth search is complete.
                     println!("    [Engine Thread] Searching depth {}...", current_depth);
                     let move_found = JamboreeSearcher::best_move(board_to_search, current_depth);
                     println!("    [Engine Thread] Depth {} search complete. Move found: {}", current_depth, move_found);

                     if move_found != BitMove::null() {
                         best_move = move_found; // Update best move found so far
                         best_move_depth = current_depth; // Record depth
                     } else {
                         // If search fails or returns null, break or use previous best
                         // This might happen at low depths if already checkmated/stalemated
                         break;
                     }

                     // Check time *after* the blocking search iteration completes
                     if Instant::now().duration_since(start_time) >= time_limit {
                         break;
                     }

                     current_depth += 1;
                }

                // Send the best move found within the time limit
                if best_move != BitMove::null() {
                     // Send move and the depth it was found at
                     println!("  [Engine Thread] Sending final move {} (found at depth {}).", best_move, best_move_depth);
                     let _ = tx.send((Some(best_move), best_move_depth));
                } else {
                    // Handle case where no move was found (should be rare if legal moves exist)
                    // Send a null move or handle appropriately
                    println!("  [Engine Thread] No legal move found or search failed. Sending null move.");
                    // Send None move with depth 0 to signal failure/completion without valid move
                    let _ = tx.send((None, 0));
                }
            });

        } else if self.state == STATE_U3 { // Engine Thinking
            if let Some(rx) = &self.rx {
                 match rx.try_recv() {
                     Ok((opt_move, depth)) => {
                         // Check if it's a final move or just a depth update
                         if let Some(applied_move) = opt_move {
                              println!("[State U3] Received FINAL move {} (depth {}) from engine thread.", applied_move, depth);

                              // --- Apply Final Move --- (This logic only runs if opt_move is Some)
                               if let Ok(mut board) = self.board.try_lock() {
                                   // Get move string *before* applying the move
                                   let move_string = applied_move.stringify(); // Use stringify()
                                   board.apply_move(applied_move);

                                   // Calculate static eval *after* move is applied
                                   println!("  -> Engine applied move. New Board FEN: {}", board.fen());
                                   let static_eval = Eval::eval_low(&board) as i16;

                                   // Update UI (TODO: Get score/depth if available from bot)
                                   self.last_move_msg = format!("Engine: {}", move_string);
                                   self.last_score_msg = format!("Score: {:.2}", static_eval as f32 / 100.0); // Show static eval
                                   self.last_depth_msg = format!("Depth: {}", depth); // Show search depth

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
                                        println!("  -> Game continues. Transitioning to UZ.");
                                   }

                                   // Clear receiver only after processing final move
                                   self.rx = None;
                                } else {
                                   ctx.request_repaint(); // Try lock again next frame
                                }
                         } else {
                            // --- Depth Update --- (This logic runs if opt_move is None)
                            if depth > 0 {
                               println!("[State U3] Received depth update: {}", depth);
                               self.last_depth_msg = format!("Depth: {} (searching...)", depth);
                               // Keep requesting repaints to check for more messages
                               ctx.request_repaint_after(Duration::from_millis(50));
                            } else {
                                // Depth 0 with None move signals failure from engine thread
                                println!("  -> Engine thread signaled failure (depth 0). Ending game.");
                                self.last_move_msg = "Engine failed to find a move.".to_owned();
                                self.state = STATE_UX; // End game?
                                self.rx = None;
                            }
                         }
                     }
                     Err(mpsc::TryRecvError::Empty) => {
                         // Engine still thinking, request repaint to check again soon
                         //println!("[State U3] Engine still thinking...");
                         ctx.request_repaint_after(Duration::from_millis(50));
                     }
                     Err(mpsc::TryRecvError::Disconnected) => {
                         // Thread panicked or sender dropped unexpectedly
                         println!("[State U3] Engine thread disconnected unexpectedly!");
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
}
