// frontend/game.js
// Live game and replay. Depends on render.js being loaded first.

const API = "http://127.0.0.1:8000";

// ── App state ──────────────────────────────────────────────────────────────

let gameState   = null;   // live GameStateResponse from API
let selection   = null;   // { source, color } | null
let lastRound   = null;

let replayRecord    = null;   // full GameRecord | null
let replayTurnIndex = 0;

let hasGameInProgress = false;  // controls whether menu close button is enabled

// ── Menu ───────────────────────────────────────────────────────────────────

function openMenu() {
  document.getElementById("menu-overlay").classList.remove("hidden");
  const closeBtn = document.getElementById("menu-close-btn");
  closeBtn.disabled = !hasGameInProgress;
  closeBtn.title = hasGameInProgress ? "Return to game" : "No game in progress";
}

function closeMenu() {
  document.getElementById("menu-overlay").classList.add("hidden");
}

function initMenu() {
  // Populate player dropdowns.
  ["player1-select", "player2-select"].forEach((id, i) => {
    const select = document.getElementById(id);
    PLAYER_OPTIONS.forEach(opt => {
      const option = document.createElement("option");
      option.value = opt.value;
      option.textContent = opt.label;
      select.appendChild(option);
    });
    select.value = i === 0 ? "human" : "greedy";
  });

  // Populate replay dropdown.
  fetchRecordingList().then(summaries => {
    const replaySelect = document.getElementById("replay-select");
    const watchBtn = document.getElementById("watch-replay-btn");
    if (summaries.length === 0) {
      replaySelect.disabled = true;
      watchBtn.disabled = true;
      const opt = document.createElement("option");
      opt.textContent = "No saved games yet";
      opt.disabled = true;
      replaySelect.appendChild(opt);
    } else {
      summaries.forEach(s => {
        const date = new Date(s.timestamp).toLocaleString();
        const scores = s.final_scores.join(" \u2013 ");
        const winnerName = s.winner !== null ? s.player_names[s.winner] : "Tie";
        const opt = document.createElement("option");
        opt.value = s.game_id;
        opt.textContent = `${date} | ${s.player_names.join(" vs ")} | ${scores} | ${winnerName}`;
        replaySelect.appendChild(opt);
      });
    }
  });

  // Wire buttons.
  document.getElementById("menu-close-btn").addEventListener("click", closeMenu);

  document.getElementById("start-game-btn").addEventListener("click", () => {
    const p1 = document.getElementById("player1-select").value;
    const p2 = document.getElementById("player2-select").value;
    closeMenu();
    startNewGame([p1, p2]);
  });

  document.getElementById("watch-replay-btn").addEventListener("click", () => {
    const gameId = document.getElementById("replay-select").value;
    if (!gameId) return;
    closeMenu();
    loadReplay(gameId);
  });
}

// ── API ────────────────────────────────────────────────────────────────────

async function fetchRecordingList() {
  const res = await fetch(`${API}/recordings`);
  return await res.json();
}

async function fetchRecord(gameId) {
  const res = await fetch(`${API}/recordings/${gameId}`);
  if (!res.ok) throw new Error(`Recording ${gameId} not found`);
  return await res.json();
}

async function loadState() {
  const res = await fetch(`${API}/state`);
  gameState = await res.json();
  selection = null;
  replayRecord = null;
  renderLive();
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
    renderLive();
    return;
  }
  gameState = await res.json();
  selection = null;
  renderLive();
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
  renderLive();
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
  replayRecord = null;
  hasGameInProgress = true;
  renderLive();
  maybeRunBot();
}

async function loadReplay(gameId) {
  replayRecord = await fetchRecord(gameId);
  replayTurnIndex = 0;
  hasGameInProgress = true;
  renderReplay();
}

// ── Bot loop ───────────────────────────────────────────────────────────────

function currentPlayerIsBot() {
  if (!gameState) return false;
  return gameState.player_types[gameState.current_player] !== "human";
}

function maybeRunBot() {
  if (!gameState || gameState.is_game_over) return;
  if (!currentPlayerIsBot()) return;
  const isNewRound = lastRound !== null && gameState.round !== lastRound;
  lastRound = gameState.round;
  setTimeout(requestAgentMove, isNewRound ? 2000 : 600);
}

// ── Selection (live game) ──────────────────────────────────────────────────

function isLegalDestination(destination) {
  if (!gameState || !selection) return false;
  return gameState.legal_moves.some(
    m => m.source === selection.source &&
         m.tile === selection.color &&
         m.destination === destination
  );
}

function handleSourceClick(source, color) {
  if (selection?.source === source && selection?.color === color) {
    selection = null;
  } else {
    selection = { source, color };
  }
  renderLive();
}

// ── Live render ────────────────────────────────────────────────────────────

function renderLive() {
  const state = gameState;
  const app = document.getElementById("app");
  app.innerHTML = "";

  // Header.
  const header = createElement("div", "header");
  header.appendChild(createElement("p", "status-line",
    state.is_game_over
      ? (state.winner !== null
          ? `Game over \u2014 Player ${state.winner + 1} wins!`
          : `Game over \u2014 it\u2019s a tie!`)
      : currentPlayerIsBot()
        ? `Player ${state.current_player + 1}\u2019s turn (bot is thinking\u2026)`
        : `Player ${state.current_player + 1}\u2019s turn`
  ));
  header.appendChild(createElement("p", "selection-hint",
    selection
      ? `Selected: ${selection.color} from ${selection.source === -1 ? "center" : `factory ${selection.source + 1}`} \u2014 click a pattern line row or the floor`
      : "\u00a0"
  ));
  const menuBtn = createElement("button", "header-btn", "Menu");
  menuBtn.addEventListener("click", openMenu);
  header.appendChild(menuBtn);
  app.appendChild(header);

  // Three-column layout: board 0, sources, board 1.
  const humanTurn = !currentPlayerIsBot() && !state.is_game_over;
  const layout = createElement("div", "game-layout");

  layout.appendChild(renderBoard(
    state.boards[0], 0,
    PLAYER_OPTIONS.find(o => o.value === state.player_types[0])?.label ?? state.player_types[0],
    state.current_player === 0,
    {
      interactive: state.current_player === 0 && humanTurn && selection !== null,
      canPlace: row => isLegalDestination(row),
      onRowClick: row => submitMove(selection.source, selection.color, row),
      canPlaceFloor: isLegalDestination(-2),
      onFloorClick: () => submitMove(selection.source, selection.color, -2),
    }
  ));

  layout.appendChild(renderSources(
    { factories: state.factories, center: state.center },
    { interactive: humanTurn, selection, onTileClick: handleSourceClick }
  ));

  layout.appendChild(renderBoard(
    state.boards[1], 1,
    PLAYER_OPTIONS.find(o => o.value === state.player_types[1])?.label ?? state.player_types[1],
    state.current_player === 1,
    {
      interactive: state.current_player === 1 && humanTurn && selection !== null,
      canPlace: row => isLegalDestination(row),
      onRowClick: row => submitMove(selection.source, selection.color, row),
      canPlaceFloor: isLegalDestination(-2),
      onFloorClick: () => submitMove(selection.source, selection.color, -2),
    }
  ));

  app.appendChild(layout);
}

// ── Replay render ──────────────────────────────────────────────────────────

function renderMoveBanner(turn) {
  const banner = createElement("div", "move-banner");
  if (!turn) {
    banner.appendChild(createElement("span", "move-text", "Start of game \u2014 no move yet."));
    return banner;
  }
  const chip = createElement("span", "move-tile-chip");
  chip.style.background = TILE_COLORS[turn.move_tile] ?? "#888";
  if (turn.move_tile === "WHITE") chip.style.border = "1px solid #ccc";
  banner.appendChild(chip);
  const playerName = replayRecord.player_names[turn.player_index];
  const src = turn.move_source === -1 ? "center" : `factory ${turn.move_source + 1}`;
  const dst = turn.move_destination === -2 ? "floor" : `row ${turn.move_destination + 1}`;
  banner.appendChild(createElement("span", "move-text",
    `${playerName} took ${turn.move_tile} from ${src} \u2192 ${dst}`));
  return banner;
}

function renderReplay() {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const record = replayRecord;
  const isAfterLastMove = replayTurnIndex >= record.turns.length;
  const currentTurn = isAfterLastMove ? null : record.turns[replayTurnIndex];

  // Board states.
  let boardStates;
  if (isAfterLastMove) {
    const lastTurn = record.turns[record.turns.length - 1];
    boardStates = lastTurn
      ? lastTurn.board_states.map((bs, i) => ({ ...bs, score: record.final_scores[i] }))
      : [];
  } else {
    boardStates = currentTurn.board_states;
  }

  const sourceState = currentTurn?.source_state ?? { factories: [], center: [] };

  // Replay controls.
  const controls = createElement("div", "replay-controls");
  const prevBtn = createElement("button", "replay-btn", "\u25c4 Prev");
  prevBtn.disabled = replayTurnIndex === 0;
  prevBtn.addEventListener("click", () => { replayTurnIndex--; renderReplay(); });
  controls.appendChild(prevBtn);

  const nextBtn = createElement("button", "replay-btn", "Next \u25ba");
  nextBtn.disabled = replayTurnIndex >= record.turns.length;
  nextBtn.addEventListener("click", () => { replayTurnIndex++; renderReplay(); });
  controls.appendChild(nextBtn);

  controls.appendChild(createElement("span", "turn-counter",
    `Move ${replayTurnIndex} / ${record.turns.length}`));

  const menuBtn = createElement("button", "replay-btn", "Menu");
  menuBtn.addEventListener("click", openMenu);
  controls.appendChild(menuBtn);

  app.appendChild(controls);
  app.appendChild(renderMoveBanner(currentTurn));
  app.appendChild(createElement("p", "status-line",
    isAfterLastMove
      ? (record.winner !== null
          ? `Game over \u2014 ${record.player_names[record.winner]} wins!`
          : "Game over \u2014 it\u2019s a tie!")
      : `${record.player_names[currentTurn.player_index]}\u2019s move`
  ));

  // Three-column layout matching live game.
  const layout = createElement("div", "game-layout");

  const makeReplayBoard = (i) => {
    const board = { ...boardStates[i], pending_placements: [], pending_bonuses: [] };
    const isActive = !isAfterLastMove && currentTurn.player_index === i;
    return renderBoard(board, i, record.player_names[i], isActive, { interactive: false });
  };

  layout.appendChild(makeReplayBoard(0));
  layout.appendChild(renderSources(sourceState, { interactive: false }));
  layout.appendChild(makeReplayBoard(1));
  app.appendChild(layout);
}

// ── Boot ───────────────────────────────────────────────────────────────────

initMenu();
openMenu();  // Show menu on load — close button starts disabled.