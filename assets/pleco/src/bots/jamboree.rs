//! The jamboree algorithm.
use super::alphabeta::alpha_beta_search;
use super::ScoringMove;
use super::*;
use board::*;
use rayon;
use std::time::{Instant, Duration};

const DIVIDE_CUTOFF: usize = 5;
const DIVISOR_SEQ: usize = 4;

pub fn jamboree(
    board: &mut Board,
    mut alpha: i16,
    beta: i16,
    depth: u16,
    plys_seq: u16,
    deadline: Option<Instant>,
) -> ScoringMove {
    assert!(alpha <= beta);
    
    // Check time constraint first
    if let Some(end_time) = deadline {
        if Instant::now() >= end_time {
            // Time's up, return best move found so far or a default move
            return ScoringMove::blank(alpha);
        }
    }
    
    if depth <= 2 {
        return alpha_beta_search(board, alpha, beta, depth);
    }

    let mut moves = board.generate_scoring_moves();

    if moves.is_empty() {
        if board.in_check() {
            return ScoringMove::blank(-MATE_V);
        } else {
            return ScoringMove::blank(DRAW_V);
        }
    }

    let amount_seq: usize = 1 + (moves.len() / DIVISOR_SEQ).min(2) as usize;
    let (seq, non_seq) = moves.split_at_mut(amount_seq);

    let mut best_move: ScoringMove = ScoringMove::blank(alpha);

    for mov in seq {
        // Check time before each move evaluation
        if let Some(end_time) = deadline {
            if Instant::now() >= end_time {
                return best_move;
            }
        }
        
        board.apply_move(mov.bit_move);
        mov.score = -jamboree(board, -beta, -alpha, depth - 1, plys_seq, deadline).score;
        board.undo_move();

        if mov.score > alpha {
            alpha = mov.score;
            if alpha >= beta {
                return *mov;
            }
            best_move = *mov;
        }
    }

    parallel_task(non_seq, board, alpha, beta, depth, plys_seq, deadline).max(best_move)
}

fn parallel_task(
    slice: &mut [ScoringMove],
    board: &mut Board,
    mut alpha: i16,
    beta: i16,
    depth: u16,
    plys_seq: u16,
    deadline: Option<Instant>,
) -> ScoringMove {
    // Check time before starting parallel task
    if let Some(end_time) = deadline {
        if Instant::now() >= end_time {
            return ScoringMove::blank(alpha);
        }
    }
    
    if slice.len() <= DIVIDE_CUTOFF {
        let mut best_move: ScoringMove = ScoringMove::blank(alpha);
        for mov in slice {
            // Check time before each move evaluation
            if let Some(end_time) = deadline {
                if Instant::now() >= end_time {
                    return best_move;
                }
            }
            
            board.apply_move(mov.bit_move);
            mov.score = -jamboree(board, -beta, -alpha, depth - 1, plys_seq, deadline).score;
            board.undo_move();
            if mov.score > alpha {
                alpha = mov.score;
                if alpha >= beta {
                    return *mov;
                }
                best_move = *mov;
            }
        }
        best_move
    } else {
        let mid_point = slice.len() / 2;
        let (left, right) = slice.split_at_mut(mid_point);
        let mut left_clone = board.parallel_clone();

        let (left_move, right_move): (ScoringMove, ScoringMove) = rayon::join(
            || parallel_task(left, &mut left_clone, alpha, beta, depth, plys_seq, deadline),
            || parallel_task(right, board, alpha, beta, depth, plys_seq, deadline),
        );

        left_move.max(right_move)
    }
}

// Helper function to check if we should terminate based on time
pub fn is_timeout(deadline: Option<Instant>) -> bool {
    if let Some(end_time) = deadline {
        return Instant::now() >= end_time;
    }
    false
}
