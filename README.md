# DrawbackChess2

A chess game with a custom egui interface for Rust enthusiasts.

![Ferris](src/ferris.png)

## Features

* **Interactive UI**: Simple but powerful egui interface with time controls and board rotation
* **Multi-threading**: Chess engine runs in a background thread to keep the UI responsive
* **Game Modes**: Support for human vs. human and engine gameplay options
* **Move Analysis**: Engine provides move scores and checkmate predictions

## Running the Game

```sh
cargo run --release
```

## Credits

This project is based on [tiny-chess](https://github.com/StefanSalewski/tiny-chess) by Dr. Stefan Salewski. The original implementation provided:

- A powerful chess engine with several enhancements over the Nim version
- A straightforward egui user interface design
- An example of threading using `spawn` and channels simultaneously

We extend our sincere thanks to Dr. Salewski for making this excellent codebase available under MIT license.

## Background

The DrawbackChess2 project builds upon the foundation of tiny-chess while introducing custom modifications to suit specific gameplay needs. The original engine serves as an excellent example of non-blocking UI design with a chess engine running in a background thread.

## License

MIT License - See LICENSE file for details. This project maintains the original MIT licensing terms from the tiny-chess project.

---

*DrawbackChess2 - Chess reimagined*
