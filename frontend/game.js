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

// Center position (in tile units) for each color's +10 bonus indicator.
// Sits between the row-4 column and the row-0 column, wrapping if needed.
const COLOR_BONUS_CENTER = {
  BLUE:   4.5,
  YELLOW: 0.5,
  RED:    1.5,
  BLACK:  2.5,
  WHITE:  3.5,
};

const FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3];

const PLAYER_OPTIONS = [
  { value: "human",     label: "Human" },
  { value: "random",    label: "Random Bot" },
  { value: "cautious",  label: "Cautious Bot" },
  { value: "efficient", label: "Efficient Bot" },
  { value: "greedy",    label: "Greedy Bot" },
  { value: "mcts",      label: "MCTS Bot" },
];

// ── State ──────────────────────────────────────────────────────────────────

let gameState = null;
let selection = null; // { source, color } or null
let lastRound = null;

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
    m => m.source      === selection.source &&
         m.tile        === selection.color  &&
         m.destination === destination
  );
}

function currentPlayerIsBot() {
  if (!gameState) return false;
  return gameState.player_types[gameState.current_player] !== "human";
}

function floorPenalty(floorLine) {
  const count = Math.min(floorLine.length, FLOOR_PENALTIES.length);
  return FLOOR_PENALTIES.slice(0, count).reduce((a, b) => a + b, 0);
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
    body: JSON.stringify({ source, tile: color, destination }),
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
  maybeRunBot();
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
  const isNewRound = lastRound !== null && gameState.round !== lastRound;
  lastRound = gameState.round;
  setTimeout(requestAgentMove, isNewRound ? 2000 : 600);
}

// ── New game config screen ─────────────────────────────────────────────────

function showNewGameDialog() {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const dialog = createElement("div", "dialog");
  dialog.appendChild(createElement("h2", "dialog-title", "New Game"));

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
    select.value = i === 0 ? "human" : "greedy";
    row.appendChild(select);
    dialog.appendChild(row);
    return select;
  });

  const startBtn = createElement("button", "start-btn", "Start");
  startBtn.addEventListener("click", () => {
    startNewGame(selects.map(s => s.value));
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

  FLOOR_PENALTIES.forEach((penalty, i) => {
    const slot = createElement("div", "floor-slot");
    slot.appendChild(createElement("div", "penalty-label", penalty));
    slot.appendChild(floorLine[i] ? makeTile(floorLine[i]) : makeTile(null));
    tilesRow.appendChild(slot);
  });

  wrapper.appendChild(tilesRow);
  return wrapper;
}

function renderWall(wall, pendingPlacements, pendingBonuses) {
  // Build a lookup of pending placements keyed by "row,column" for fast access.
  const pendingMap = {};
  pendingPlacements.forEach(p => { pendingMap[`${p.row},${p.column}`] = p.placement_points; });

  // Build a lookup of pending row bonuses keyed by row index.
  const rowBonusMap = {};
  pendingBonuses
    .filter(b => b.bonus_type === "row")
    .forEach(b => { rowBonusMap[b.index] = b.bonus_points; });

  // Build a lookup of pending column bonuses keyed by column index.
  const columnBonusMap = {};
  pendingBonuses
    .filter(b => b.bonus_type === "column")
    .forEach(b => { columnBonusMap[b.index] = b.bonus_points; });

  // Build a lookup of pending color bonuses keyed by color name.
  const colorBonusMap = {};
  pendingBonuses
    .filter(b => b.bonus_type === "tile")
    .forEach(b => {
      // Map Tile enum value back to color name using WALL_PATTERN row 0.
      // Tile enum: BLUE=1, YELLOW=2, RED=3, BLACK=4, WHITE=5.
      const colorName = ["BLUE", "YELLOW", "RED", "BLACK", "WHITE"][b.index - 1];
      colorBonusMap[colorName] = b.bonus_points;
    });

  const TILE_SIZE = 36;  // px — must match .tile width/height in CSS
  const GAP = 4;         // px — must match wall-row gap in CSS
  const TILE_STEP = TILE_SIZE + GAP; // px per tile unit

  // Outer wrapper: wall grid on the left, row bonus column on the right.
  const wrapper = createElement("div", "wall-wrapper");

  // Row: wall grid + row bonus column side by side.
  const wallAndBonuses = createElement("div", "wall-and-row-bonuses");

  const wallGrid = createElement("div", "wall");
  wall.forEach((row, rowIndex) => {
    const rowEl = createElement("div", "wall-row");
    row.forEach((cell, colIndex) => {
      const key = `${rowIndex},${colIndex}`;
      if (key in pendingMap) {
        const tileColor = WALL_PATTERN[rowIndex][colIndex];
        const tileEl = makeTile(tileColor, false);
        tileEl.classList.add("pending-tile");
        const label = createElement("div", "pending-label", `+${pendingMap[key]}`);
        tileEl.appendChild(label);
        rowEl.appendChild(tileEl);
      } else {
        rowEl.appendChild(
          cell ? makeTile(cell) : makeTile(WALL_PATTERN[rowIndex][colIndex], true)
        );
      }
    });
    wallGrid.appendChild(rowEl);
  });
  wallAndBonuses.appendChild(wallGrid);

  // Row bonus column — one slot per wall row, aligned to each row.
  const rowBonusCol = createElement("div", "row-bonus-col");
  wall.forEach((_, rowIndex) => {
    const slot = createElement("div", "row-bonus-slot");
    if (rowIndex in rowBonusMap) {
      slot.appendChild(createElement("span", "bonus-label", `+${rowBonusMap[rowIndex]}`));
    }
    rowBonusCol.appendChild(slot);
  });
  wallAndBonuses.appendChild(rowBonusCol);
  wrapper.appendChild(wallAndBonuses);

  // Below-wall strip: column bonuses and color bonuses.
  const belowStrip = createElement("div", "below-wall-strip");
  belowStrip.style.width = `${5 * TILE_STEP - GAP}px`;
  belowStrip.style.position = "relative";

  // Column bonuses — centered on each column.
  Object.entries(columnBonusMap).forEach(([colIndex, points]) => {
    const label = createElement("span", "bonus-label", `+${points}`);
    label.style.position = "absolute";
    label.style.left = `${colIndex * TILE_STEP + TILE_SIZE / 2}px`;
    label.style.transform = "translateX(-50%)";
    belowStrip.appendChild(label);
  });

  // Color bonuses — centered at the defined half-tile positions.
  Object.entries(colorBonusMap).forEach(([colorName, points]) => {
    const center = COLOR_BONUS_CENTER[colorName]; // e.g. 0.5, 1.5, ... 4.5
    const N = Math.floor(center);                 // left column of the pair
    const px = N * TILE_STEP + TILE_STEP / 2 + TILE_SIZE / 2;
    const label = createElement("span", "bonus-label bonus-color", `+${points}`);
    label.style.position = "absolute";
    label.style.left = `${px}px`;
    label.style.transform = "translateX(-50%)";
    belowStrip.appendChild(label);
  });

  // Only add the strip if there is something to show.
  if (Object.keys(columnBonusMap).length > 0 || Object.keys(colorBonusMap).length > 0) {
    wrapper.appendChild(belowStrip);
  }

  return wrapper;
}

function renderScoreDisplay(board) {
  const penalty = floorPenalty(board.floor_line);
  const placementPoints = board.pending_placements.reduce(
    (sum, p) => sum + p.placement_points, 0
  );
  const bonusPoints = board.pending_bonuses.reduce(
    (sum, b) => sum + b.bonus_points, 0
  );
  const grandTotal = board.score + placementPoints + penalty + bonusPoints;

  const bar = createElement("div", "score-bar");

  const addPart = (text, className) => {
    bar.appendChild(createElement("span", `score-part ${className}`, text));
  };

  addPart(`${board.score}`, "score-carried");
  addPart(` + ${placementPoints}`, "score-pending");
  // Floor penalty: always shown, use − sign when negative, + when zero.
  addPart(penalty < 0 ? ` − ${Math.abs(penalty)}` : ` + 0`, "score-floor");
  addPart(` + ${bonusPoints}`, "score-bonus");
  addPart(` = ${grandTotal}`, "score-total");

  return bar;
}

function renderBoard(board, index, isActive) {
  const playerType = gameState?.player_types?.[index] ?? "human";
  const label = PLAYER_OPTIONS.find(o => o.value === playerType)?.label ?? playerType;

  const wrapper = createElement("div",
    `player-board${isActive ? " active-player" : ""}`);

  const heading = createElement("div", "player-heading");
  heading.appendChild(
    createElement("span", "player-name", `Player ${index + 1} (${label})`)
  );
  heading.appendChild(renderScoreDisplay(board));
  wrapper.appendChild(heading);

  const middle = createElement("div", "board-middle");
  middle.appendChild(renderPatternLines(board.pattern_lines, isActive));
  middle.appendChild(createElement("div", "board-divider"));
  middle.appendChild(renderWall(board.wall, board.pending_placements, board.pending_bonuses));
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
      ? (state.winner !== null
          ? `Game over — Player ${state.winner + 1} wins!`
          : `Game over — it's a tie!`)
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