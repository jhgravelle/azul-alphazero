// frontend/replay.js

const API = "http://127.0.0.1:8000";

// Shared constants — identical to game.js.
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

const FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3];

// ── State ──────────────────────────────────────────────────────────────────

let record = null;   // full GameRecord loaded from API
let turnIndex = 0;   // which turn we are currently viewing (0 = before any moves)

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

function floorPenalty(floorLine) {
  const count = Math.min(floorLine.length, FLOOR_PENALTIES.length);
  return FLOOR_PENALTIES.slice(0, count).reduce((a, b) => a + b, 0);
}

function sourceLabel(source) {
  return source === -1 ? "center" : `factory ${source + 1}`;
}

function destinationLabel(destination) {
  return destination === -2 ? "floor" : `row ${destination + 1}`;
}

// ── API ────────────────────────────────────────────────────────────────────

async function fetchRecordingList() {
  const res = await fetch(`${API}/recordings`);
  return await res.json(); // list of RecordingSummary
}

async function fetchRecord(gameId) {
  const res = await fetch(`${API}/recordings/${gameId}`);
  if (!res.ok) throw new Error(`Recording ${gameId} not found`);
  return await res.json();
}

// ── Board rendering ────────────────────────────────────────────────────────
// These functions accept the raw board_state dict from the recording
// (no pending placements / bonuses — just score, wall, pattern_lines, floor_line).

function renderPatternLines(patternLines) {
  const wrapper = createElement("div", "pattern-lines");
  patternLines.forEach((line, row) => {
    const rowEl = createElement("div", "pattern-row");
    const capacity = row + 1;
    const emptyCount = capacity - line.length;
    for (let i = 0; i < emptyCount; i++) rowEl.appendChild(makeTile(null));
    for (const color of line) rowEl.appendChild(makeTile(color));
    wrapper.appendChild(rowEl);
  });
  return wrapper;
}

function renderWall(wall) {
  const wallGrid = createElement("div", "wall");
  wall.forEach((row, rowIndex) => {
    const rowEl = createElement("div", "wall-row");
    row.forEach((cell, colIndex) => {
      rowEl.appendChild(
        cell ? makeTile(cell) : makeTile(WALL_PATTERN[rowIndex][colIndex], true)
      );
    });
    wallGrid.appendChild(rowEl);
  });
  return wallGrid;
}

function renderFloorLine(floorLine) {
  const wrapper = createElement("div", "floor-line");
  wrapper.appendChild(createElement("div", "pool-label", "Floor"));
  const tilesRow = createElement("div", "floor-tiles");
  FLOOR_PENALTIES.forEach((penalty, i) => {
    const slot = createElement("div", "floor-slot");
    slot.appendChild(createElement("div", "penalty-label", penalty));
    slot.appendChild(floorLine[i] ? makeTile(floorLine[i]) : makeTile(null));
    tilesRow.appendChild(slot);
  });
  wrapper.appendChild(tilesRow);
  return wrapper;
}

function renderScoreBar(boardState) {
  const penalty = floorPenalty(boardState.floor_line);
  const bar = createElement("div", "score-bar");
  bar.appendChild(createElement("span", "score-carried", `${boardState.score}`));
  bar.appendChild(createElement("span", "score-floor",
    penalty < 0 ? ` − ${Math.abs(penalty)} pending` : ``));
  return bar;
}

function renderBoard(boardState, playerIndex, playerName, isActive) {
  const wrapper = createElement("div",
    `player-board${isActive ? " active-player" : ""}`);

  const heading = createElement("div", "player-heading");
  heading.appendChild(createElement("span", "player-name",
    `Player ${playerIndex + 1} (${playerName})`));
  heading.appendChild(renderScoreBar(boardState));
  wrapper.appendChild(heading);

  const middle = createElement("div", "board-middle");
  middle.appendChild(renderPatternLines(boardState.pattern_lines));
  middle.appendChild(createElement("div", "board-divider"));
  middle.appendChild(renderWall(boardState.wall));
  wrapper.appendChild(middle);
  wrapper.appendChild(renderFloorLine(boardState.floor_line));
  return wrapper;
}

// ── Move banner ────────────────────────────────────────────────────────────

function renderMoveBanner(turn) {
  const banner = createElement("div", "move-banner");

  if (!turn) {
    banner.appendChild(createElement("span", "move-text",
      "Start of game — no move yet."));
    return banner;
  }

  const chip = createElement("span", "move-tile-chip");
  chip.style.background = TILE_COLORS[turn.move_tile] ?? "#888";
  if (turn.move_tile === "WHITE") chip.style.border = "1px solid #ccc";
  banner.appendChild(chip);

  const playerName = record.player_names[turn.player_index];
  banner.appendChild(createElement("span", "move-text",
    `${playerName} took ${turn.move_tile} from ${sourceLabel(turn.move_source)} → ${destinationLabel(turn.move_destination)}`
  ));

  return banner;
}

// ── Main render ────────────────────────────────────────────────────────────

function renderReplay() {
  const app = document.getElementById("app");
  app.innerHTML = "";

  // If no record is loaded yet, show only the controls.
  app.appendChild(renderControls());

  if (!record) return;

  // The turn we display is the board state captured BEFORE this move was made.
  // turnIndex 0 means "before move 0" — the initial board state.
  // turnIndex record.turns.length means "after the last move" — use final_scores.
  const isAfterLastMove = turnIndex >= record.turns.length;
  const currentTurn = isAfterLastMove ? null : record.turns[turnIndex];

  // Board states: use the snapshot from the current turn if mid-game,
  // or the final scores if we're at the end.
  let boardStates;
  if (isAfterLastMove) {
    // Reconstruct a minimal board state from the last turn's snapshot
    // but with final scores applied.
    const lastTurn = record.turns[record.turns.length - 1];
    boardStates = lastTurn
      ? lastTurn.board_states.map((bs, i) => ({
          ...bs,
          score: record.final_scores[i],
        }))
      : record.turns[0]?.board_states ?? [];
  } else {
    boardStates = currentTurn.board_states;
  }

  // Move banner — shows the move that is ABOUT TO BE played (i.e. currentTurn).
  app.appendChild(renderMoveBanner(currentTurn));

  // Status line.
  const status = createElement("p", "status-line",
    isAfterLastMove
      ? (record.winner !== null
          ? `Game over — ${record.player_names[record.winner]} wins!`
          : "Game over — it's a tie!")
      : `${record.player_names[currentTurn.player_index]}'s move`
  );
  app.appendChild(status);

  // Boards.
  const boards = createElement("div", "boards");
  boardStates.forEach((boardState, i) => {
    const isActive = !isAfterLastMove && currentTurn.player_index === i;
    boards.appendChild(
      renderBoard(boardState, i, record.player_names[i], isActive)
    );
  });
  app.appendChild(boards);
}

// ── Controls ───────────────────────────────────────────────────────────────

function renderControls() {
  // Preserve the existing select element if it's already in the DOM
  // so the dropdown selection survives re-renders.
  const existing = document.querySelector(".replay-select");
  const controls = createElement("div", "replay-controls");

  const select = existing ?? createElement("select", "replay-select");
  if (!existing) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select a saved game…";
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);
    select.addEventListener("change", () => loadRecording(select.value));
  }
  controls.appendChild(select);

  const total = record ? record.turns.length : 0;

  const prevBtn = createElement("button", "replay-btn", "◀ Prev");
  prevBtn.disabled = !record || turnIndex === 0;
  prevBtn.addEventListener("click", () => { turnIndex--; renderReplay(); });
  controls.appendChild(prevBtn);

  const nextBtn = createElement("button", "replay-btn", "Next ▶");
  nextBtn.disabled = !record || turnIndex >= total;
  nextBtn.addEventListener("click", () => { turnIndex++; renderReplay(); });
  controls.appendChild(nextBtn);

  const counter = createElement("span", "turn-counter",
    record ? `Move ${turnIndex} / ${total}` : "");
  controls.appendChild(counter);

  const homeLink = createElement("a", "replay-btn", "← Live game");
  homeLink.href = "index.html";
  homeLink.style.textDecoration = "none";
  controls.appendChild(homeLink);

  return controls;
}

// ── Loading ────────────────────────────────────────────────────────────────

async function loadRecording(gameId) {
  record = await fetchRecord(gameId);
  turnIndex = 0;
  renderReplay();
}

async function populateDropdown() {
  const summaries = await fetchRecordingList();
  const app = document.getElementById("app");

  if (summaries.length === 0) {
    app.innerHTML = "";
    app.appendChild(renderControls());
    app.appendChild(createElement("p", "replay-empty",
      "No saved games yet. Play a game first!"));
    return;
  }

  app.innerHTML = "";
  app.appendChild(renderControls());

  // Now that the controls are in the DOM, populate the select.
  const select = document.querySelector(".replay-select");
  summaries.forEach(s => {
    const date = new Date(s.timestamp).toLocaleString();
    const scores = s.final_scores.join(" – ");
    const winnerName = s.winner !== null ? s.player_names[s.winner] : "Tie";
    const option = document.createElement("option");
    option.value = s.game_id;
    option.textContent = `${date} | ${s.player_names.join(" vs ")} | ${scores} | ${winnerName}`;
    select.appendChild(option);
  });
}

// ── Boot ───────────────────────────────────────────────────────────────────

populateDropdown();