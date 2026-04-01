// frontend/game.js

const API = "http://127.0.0.1:8000";

const TILE_COLORS = {
  BLUE:         "#3a7abf",
  YELLOW:       "#e8c840",
  RED:          "#c0392b",
  BLACK:        "#2c2c2c",
  WHITE:        "#f0f0f0",
  FIRST_PLAYER: "#c0a0d0",
};

const WALL_PATTERN = [
  ["BLUE",   "YELLOW", "RED",    "BLACK",  "WHITE"],
  ["WHITE",  "BLUE",   "YELLOW", "RED",    "BLACK"],
  ["BLACK",  "WHITE",  "BLUE",   "YELLOW", "RED"  ],
  ["RED",    "BLACK",  "WHITE",  "BLUE",   "YELLOW"],
  ["YELLOW", "RED",    "BLACK",  "WHITE",  "BLUE" ],
];

// ── State ──────────────────────────────────────────────────────────────────

let gameState = null;
let selection = null; // { source, color } or null

// ── Helpers ────────────────────────────────────────────────────────────────

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function makeTile(color, faint = false) {
  const tile = createElement("div", "tile");
  if (color) {
    tile.style.background = TILE_COLORS[color];
    if (color === "WHITE") tile.style.border = "1px solid #ccc";
  } else {
    tile.style.background = "#d0c8b8";
  }
  if (faint) tile.style.opacity = "0.25";
  return tile;
}

function isLegalDestination(destination) {
  if (!gameState || !selection) return false;
  return gameState.legal_moves.some(
    move =>
      move.source === selection.source &&
      move.color === selection.color &&
      move.destination === destination
  );
}

// ── API calls ──────────────────────────────────────────────────────────────

async function loadState() {
  const response = await fetch(`${API}/state`);
  gameState = await response.json();
  selection = null;
  render(gameState);
}

async function submitMove(source, color, destination) {
  const response = await fetch(`${API}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, color, destination }),
  });
  if (!response.ok) {
    const error = await response.json();
    console.error("Illegal move:", error.detail);
    selection = null;
    render(gameState);
    return;
  }
  gameState = await response.json();
  selection = null;
  render(gameState);
}

async function startNewGame() {
  const response = await fetch(`${API}/new-game`, { method: "POST" });
  gameState = await response.json();
  selection = null;
  render(gameState);
}

// ── Selection ──────────────────────────────────────────────────────────────

function handleSourceClick(source, color) {
  if (selection?.source === source && selection?.color === color) {
    selection = null; // reclick to deselect
  } else {
    selection = { source, color };
  }
  render(gameState);
}

// ── Factories and center ───────────────────────────────────────────────────

function renderSources(state) {
  const section = createElement("section", "sources");

  state.factories.forEach((factory, index) => {
    const display = createElement("div", "factory");
    const grid = createElement("div", "factory-grid");

    for (let i = 0; i < 4; i++) {
      const color = factory[i] ?? null;
      const tile = makeTile(color);

      if (color && color !== "FIRST_PLAYER" && !state.is_game_over) {
        tile.classList.add("clickable");
        if (selection?.source === index && selection?.color === color) {
          tile.classList.add("selected");
        }
        tile.addEventListener("click", () => handleSourceClick(index, color));
      }

      grid.appendChild(tile);
    }

    display.appendChild(grid);
    section.appendChild(display);
  });

  // Center pool
  const centerPool = createElement("div", "center-pool");
  centerPool.appendChild(createElement("div", "pool-label", "Center"));
  const centerTiles = createElement("div", "center-tiles");

  if (state.center.length === 0) {
    centerTiles.appendChild(createElement("span", "empty-label", "empty"));
  } else {
    state.center.forEach(color => {
      const tile = makeTile(color);
      if (color !== "FIRST_PLAYER" && !state.is_game_over) {
        tile.classList.add("clickable");
        if (selection?.source === -1 && selection?.color === color) {
          tile.classList.add("selected");
        }
        tile.addEventListener("click", () => handleSourceClick(-1, color));
      }
      centerTiles.appendChild(tile);
    });
  }

  centerPool.appendChild(centerTiles);
  section.appendChild(centerPool);

  return section;
}

// ── Player board ───────────────────────────────────────────────────────────

function renderPatternLines(patternLines, isActive) {
  const wrapper = createElement("div", "pattern-lines");

  patternLines.forEach((line, row) => {
    const rowElement = createElement("div", "pattern-row");
    const capacity = row + 1;
    const canPlace = isActive && selection !== null && isLegalDestination(row);

    if (canPlace) {
      rowElement.classList.add("clickable-row");
      rowElement.addEventListener("click", () =>
        submitMove(selection.source, selection.color, row)
      );
    }

    // Fill right to left: empty slots on the left, tiles on the right
    const emptyCount = capacity - line.length;
    for (let i = 0; i < emptyCount; i++) {
      rowElement.appendChild(makeTile(null));
    }
    for (let i = 0; i < line.length; i++) {
      rowElement.appendChild(makeTile(line[i]));
    }

    wrapper.appendChild(rowElement);
  });

  return wrapper;
}

function renderFloorLine(floorLine, isActive) {
  const wrapper = createElement("div", "floor-line");
  const penalties = [-1, -1, -2, -2, -2, -3, -3];

  wrapper.appendChild(createElement("div", "pool-label", "Floor"));

  const tilesRow = createElement("div", "floor-tiles");
  const canPlaceOnFloor = isActive && selection !== null && isLegalDestination(-2);

  if (canPlaceOnFloor) {
    tilesRow.classList.add("clickable-row");
    tilesRow.addEventListener("click", () =>
      submitMove(selection.source, selection.color, -2)
    );
  }

  penalties.forEach((penalty, i) => {
    const slot = createElement("div", "floor-slot");
    slot.appendChild(createElement("div", "penalty-label", penalty));
    slot.appendChild(floorLine[i] ? makeTile(floorLine[i]) : makeTile(null));
    tilesRow.appendChild(slot);
  });

  wrapper.appendChild(tilesRow);
  return wrapper;
}

function renderWall(wall) {
  const wrapper = createElement("div", "wall");
  wall.forEach((row, rowIndex) => {
    const rowElement = createElement("div", "wall-row");
    row.forEach((cell, colIndex) => {
      rowElement.appendChild(
        cell
          ? makeTile(cell)
          : makeTile(WALL_PATTERN[rowIndex][colIndex], true)
      );
    });
    wrapper.appendChild(rowElement);
  });
  return wrapper;
}

function renderBoard(board, index, isActive) {
  const wrapper = createElement("div",
    `player-board${isActive ? " active-player" : ""}`);

  wrapper.appendChild(createElement("h2", "player-heading",
    `Player ${index + 1} — Score: ${board.score}`));

  const middle = createElement("div", "board-middle");
  middle.appendChild(renderPatternLines(board.pattern_lines, isActive));
  middle.appendChild(createElement("div", "board-divider"));
  middle.appendChild(renderWall(board.wall));
  wrapper.appendChild(middle);

  wrapper.appendChild(renderFloorLine(board.floor_line, isActive));
  return wrapper;
}

// ── Top-level render ───────────────────────────────────────────────────────

function render(state) {
  const app = document.getElementById("app");
  app.innerHTML = "";

  // Header
  const header = createElement("div", "header");

  header.appendChild(createElement("p", "status-line",
    state.is_game_over
      ? `Game over — Player ${state.winner + 1} wins!`
      : `Player ${state.current_player + 1}'s turn`
  ));

  // Reserve space for hint so layout doesn't jump
  const hint = createElement("p", "selection-hint",
    selection
      ? `Selected: ${selection.color} from ${selection.source === -1 ? "center" : `factory ${selection.source + 1}`} — click a pattern line row or the floor`
      : " " // non-breaking space holds the line height
  );
  header.appendChild(hint);

  const newGameButton = createElement("button", "new-game-btn", "New Game");
  newGameButton.addEventListener("click", startNewGame);
  header.appendChild(newGameButton);

  app.appendChild(header);
  app.appendChild(renderSources(state));

  const boards = createElement("div", "boards");
  state.boards.forEach((board, index) => {
    boards.appendChild(
      renderBoard(board, index, index === state.current_player)
    );
  });
  app.appendChild(boards);
}

// ── Boot ───────────────────────────────────────────────────────────────────

loadState();