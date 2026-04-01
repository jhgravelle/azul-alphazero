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
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
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
    m => m.source === selection.source &&
         m.color  === selection.color  &&
         m.destination === destination
  );
}

function currentPlayerIsBot() {
  if (!gameState) return false;
  return gameState.player_types[gameState.current_player] !== "human";
}

// ── API calls ──────────────────────────────────────────────────────────────

async function loadState() {
  const res = await fetch(`${API}/state`);
  gameState = await res.json();
  selection = null;
  render(gameState);
  maybeRunBot();
}

async function submitMove(source, color, destination) {
  const res = await fetch(`${API}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, color, destination }),
  });
  if (!res.ok) {
    const err = await res.json();
    console.error("Illegal move:", err.detail);
    selection = null;
    render(gameState);
    return;
  }
  gameState = await res.json();
  selection = null;
  render(gameState);
  maybeRunBot();
}

async function requestAgentMove() {
  const res = await fetch(`${API}/agent-move`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json();
    console.error("Agent move failed:", err.detail);
    return;
  }
  gameState = await res.json();
  selection = null;
  render(gameState);
  maybeRunBot(); // chain — in case both players are bots
}

async function startNewGame(playerTypes) {
  const res = await fetch(`${API}/new-game`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_types: playerTypes }),
  });
  gameState = await res.json();
  selection = null;
  render(gameState);
  maybeRunBot();
}

// ── Bot loop ───────────────────────────────────────────────────────────────

function maybeRunBot() {
  if (!gameState || gameState.is_game_over) return;
  if (!currentPlayerIsBot()) return;
  // Small delay so the human can see the board before the bot moves
  setTimeout(requestAgentMove, 600);
}

// ── New game config screen ─────────────────────────────────────────────────

function showNewGameDialog() {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const dialog = createElement("div", "dialog");
  dialog.appendChild(createElement("h2", "dialog-title", "New Game"));

  const PLAYER_OPTIONS = [
    { value: "human",  label: "Human" },
    { value: "random", label: "Random Bot" },
  ];

  const selects = [0, 1].map(i => {
    const row = createElement("div", "dialog-row");
    row.appendChild(createElement("label", "dialog-label", `Player ${i + 1}`));
    const select = document.createElement("select");
    select.className = "dialog-select";
    PLAYER_OPTIONS.forEach(opt => {
      const option = document.createElement("option");
      option.value = opt.value;
      option.textContent = opt.label;
      select.appendChild(option);
    });
    // Default: P1 human, P2 random
    select.value = i === 0 ? "human" : "random";
    row.appendChild(select);
    dialog.appendChild(row);
    return select;
  });

  const startBtn = createElement("button", "start-btn", "Start");
  startBtn.addEventListener("click", () => {
    const playerTypes = selects.map(s => s.value);
    startNewGame(playerTypes);
  });
  dialog.appendChild(startBtn);

  app.appendChild(dialog);
}

// ── Selection ──────────────────────────────────────────────────────────────

function handleSourceClick(source, color) {
  if (selection?.source === source && selection?.color === color) {
    selection = null;
  } else {
    selection = { source, color };
  }
  render(gameState);
}

// ── Factories and center ───────────────────────────────────────────────────

function renderSources(state) {
  const section = createElement("section", "sources");
  const humanTurn = !currentPlayerIsBot() && !state.is_game_over;

  state.factories.forEach((factory, index) => {
    const display = createElement("div", "factory");
    const grid = createElement("div", "factory-grid");
    for (let i = 0; i < 4; i++) {
      const color = factory[i] ?? null;
      const tile = makeTile(color);
      if (color && color !== "FIRST_PLAYER" && humanTurn) {
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

  const centerPool = createElement("div", "center-pool");
  centerPool.appendChild(createElement("div", "pool-label", "Center"));
  const centerTiles = createElement("div", "center-tiles");

  if (state.center.length === 0) {
    centerTiles.appendChild(createElement("span", "empty-label", "empty"));
  } else {
    state.center.forEach(color => {
      const tile = makeTile(color);
      if (color !== "FIRST_PLAYER" && humanTurn) {
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
  const humanTurn = !currentPlayerIsBot();

  patternLines.forEach((line, row) => {
    const rowEl = createElement("div", "pattern-row");
    const capacity = row + 1;
    const canPlace = isActive && humanTurn && selection !== null && isLegalDestination(row);

    if (canPlace) {
      rowEl.classList.add("clickable-row");
      rowEl.addEventListener("click", () =>
        submitMove(selection.source, selection.color, row)
      );
    }

    const emptyCount = capacity - line.length;
    for (let i = 0; i < emptyCount; i++) rowEl.appendChild(makeTile(null));
    for (let i = 0; i < line.length; i++) rowEl.appendChild(makeTile(line[i]));

    wrapper.appendChild(rowEl);
  });

  return wrapper;
}

function renderFloorLine(floorLine, isActive) {
  const wrapper = createElement("div", "floor-line");
  const penalties = [-1, -1, -2, -2, -2, -3, -3];
  const humanTurn = !currentPlayerIsBot();

  wrapper.appendChild(createElement("div", "pool-label", "Floor"));

  const tilesRow = createElement("div", "floor-tiles");
  const canPlace = isActive && humanTurn && selection !== null && isLegalDestination(-2);

  if (canPlace) {
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
    const rowEl = createElement("div", "wall-row");
    row.forEach((cell, colIndex) => {
      rowEl.appendChild(
        cell ? makeTile(cell) : makeTile(WALL_PATTERN[rowIndex][colIndex], true)
      );
    });
    wrapper.appendChild(rowEl);
  });
  return wrapper;
}

function renderBoard(board, index, isActive) {
  const playerType = gameState?.player_types?.[index] ?? "human";
  const label = playerType === "human" ? "Human" : "Random Bot";

  const wrapper = createElement("div",
    `player-board${isActive ? " active-player" : ""}`);
  wrapper.appendChild(createElement("h2", "player-heading",
    `Player ${index + 1} (${label}) — Score: ${board.score}`));

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

  const header = createElement("div", "header");

  header.appendChild(createElement("p", "status-line",
    state.is_game_over
      ? `Game over — Player ${state.winner + 1} wins!`
      : currentPlayerIsBot()
        ? `Player ${state.current_player + 1}'s turn (bot is thinking…)`
        : `Player ${state.current_player + 1}'s turn`
  ));

  const hint = createElement("p", "selection-hint",
    selection
      ? `Selected: ${selection.color} from ${selection.source === -1 ? "center" : `factory ${selection.source + 1}`} — click a pattern line row or the floor`
      : " "
  );
  header.appendChild(hint);

  const newGameBtn = createElement("button", "new-game-btn", "New Game");
  newGameBtn.addEventListener("click", showNewGameDialog);
  header.appendChild(newGameBtn);

  app.appendChild(header);
  app.appendChild(renderSources(state));

  const boards = createElement("div", "boards");
  state.boards.forEach((board, index) => {
    boards.appendChild(renderBoard(board, index, index === state.current_player));
  });
  app.appendChild(boards);
}

// ── Boot ───────────────────────────────────────────────────────────────────

loadState();