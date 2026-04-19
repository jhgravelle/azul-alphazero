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

// ── Hypothetical tree state ────────────────────────────────────────────────

let _hypRoot    = null;   // HypNode — root of the exploration tree
let _hypCurrent = null;   // HypNode — currently displayed node
let _hypNodeId  = 0;      // auto-increment for unique node IDs
let _hypOriginalPlayerTypes = null;  // real player types before hyp mode overrode them

function _makeNode(parent, move, state) {
  const scores = state.boards.map(board => {
    const pending = (board.pending_placements ?? []).reduce(
      (sum, p) => sum + p.placement_points, 0);
    const bonuses = (board.pending_bonuses ?? []).reduce(
      (sum, b) => sum + b.bonus_points, 0);
    const penalty = floorPenalty(board.floor_line);
    return board.score + pending + bonuses + penalty;
  });
  return {
    id: _hypNodeId++,
    parent,
    children: [],
    move,
    scores,
    state,
    isCurrent: true,
  };
}

function _setCurrentNode(node) {
  if (_hypCurrent) _hypCurrent.isCurrent = false;
  _hypCurrent = node;
  node.isCurrent = true;
}

function _recordHypNode(move, newState) {
  const node = _makeNode(_hypCurrent, move, newState);
  _hypCurrent.children.push(node);
  _setCurrentNode(node);
}

function _countInSource(state, source, color) {
  if (source === -1) return state.center.filter(t => t === color).length;
  return (state.factories[source] ?? []).filter(t => t === color).length;
}

// ── Inspector state ────────────────────────────────────────────────────────

let _inspectorEnabled   = false;
let _inspectorSnapshot  = null;
let _inspectorSSE       = null;   // EventSource | null
let _inspectorExpanded  = new Set();
let _showPerspective    = false;  // hook: set to true to show perspective label

function _inspectorToggle() {
  _inspectorEnabled = !_inspectorEnabled;
  if (!_inspectorEnabled) {
    _inspectorCloseSSE();
  } else {
    _inspectorConnect();
  }
  _renderInspector();
}

function _inspectorCloseSSE() {
  if (_inspectorSSE) {
    _inspectorSSE.close();
    _inspectorSSE = null;
  }
}

function _inspectorConnect() {
  _inspectorCloseSSE();
  _inspectorSnapshot = null;

  const url = _inspectorStreamURL();
  if (!url) return;

  const sse = new EventSource(url);
  _inspectorSSE = sse;

  sse.addEventListener("update", e => {
    _inspectorSnapshot = JSON.parse(e.data);
    _renderInspector();
    if (_inspectorSnapshot.done) {
      sse.close();
      _inspectorSSE = null;
    }
  });

  sse.onerror = () => {
    sse.close();
    _inspectorSSE = null;
  };
}

function _inspectorStreamURL() {
  if (replayRecord) {
    // Replay mode: use game_id + flat move index
    const gameId = replayRecord.game_id;
    const moveIndex = replayTurnIndex;
    return `${API}/inspect/${gameId}/${moveIndex}/stream`;
  }
  if (gameState) {
    // Live mode: POST snapshot first, then connect to SSE
    _inspectorPostLive().then(() => {
      if (!_inspectorEnabled) return;
      const sse = new EventSource(`${API}/inspect/live/stream`);
      _inspectorSSE = sse;
      sse.addEventListener("update", e => {
        _inspectorSnapshot = JSON.parse(e.data);
        _renderInspector();
        if (_inspectorSnapshot.done) {
          sse.close();
          _inspectorSSE = null;
        }
      });
      sse.onerror = () => { sse.close(); _inspectorSSE = null; };
    });
    return null; // SSE set up async above
  }
  return null;
}

async function _inspectorPostLive() {
  if (!gameState) return;
  const snapshot = {
    current_player: gameState.current_player,
    factories: gameState.factories,
    center: gameState.center,
    boards: gameState.boards.map(b => ({
      score: b.score,
      wall: b.wall,
      pattern_lines: b.pattern_lines,
      floor_line: b.floor_line,
    })),
  };
  const res = await fetch(`${API}/inspect/live`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(snapshot),
  });
  if (res.ok) {
    _inspectorSnapshot = await res.json();
    _renderInspector();
  }
}

async function _inspectorExtend() {
  const res = await fetch(`${API}/inspect/extend`, { method: "POST" });
  if (!res.ok) return;
  _inspectorSnapshot = await res.json();
  _renderInspector();
  // If not yet stable, reconnect SSE to keep streaming
  if (!_inspectorSnapshot.done && !_inspectorSSE) {
    _inspectorConnect();
  }
}

function _inspectorNodeToggle(key) {
  if (_inspectorExpanded.has(key)) {
    _inspectorExpanded.delete(key);
  } else {
    _inspectorExpanded.add(key);
  }
  _renderInspector();
}

function _renderInspector() {
  // Re-render only the inspector panel in place.
  // Find or create the inspector mount point.
  const app = document.getElementById("app");
  let mount = document.getElementById("inspector-mount");
  if (!mount) {
    mount = createElement("div", "");
    mount.id = "inspector-mount";
    app.appendChild(mount);
  }
  mount.innerHTML = "";
  mount.appendChild(renderInspectorPanel(_inspectorSnapshot, {
    enabled: _inspectorEnabled,
    showPerspective: _showPerspective,
    onToggle: _inspectorToggle,
    onExtend: _inspectorExtend,
    onNodeToggle: _inspectorNodeToggle,
    expandedKeys: _inspectorExpanded,
  }));
}

function _inspectorReset() {
  _inspectorCloseSSE();
  _inspectorSnapshot = null;
  _inspectorExpanded = new Set();
  // Don't reset _inspectorEnabled — keep panel open across navigation
}

// ── Menu ───────────────────────────────────────────────────────────────────

function _makeMenuButton(cssClass = "header-btn secondary") {
  const menuBtn = createElement("button", cssClass, "Menu");
  menuBtn.addEventListener("click", openMenu);
  return menuBtn;
}

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
      const groups = { eval: [], human: [] };
      summaries.forEach(s => {
        const bucket = groups[s.folder] ?? groups.human;
        bucket.push(s);
      });

      const addGroup = (label, items) => {
        if (items.length === 0) return;
        const sep = document.createElement("option");
        sep.textContent = `── ${label} ──`;
        sep.disabled = true;
        replaySelect.appendChild(sep);
        items.forEach(s => {
          const date = new Date(s.timestamp).toLocaleString();
          const scores = s.final_scores.join(" \u2013 ");
          const winnerName = s.winner !== null ? s.player_names[s.winner] : "Tie";
          const opt = document.createElement("option");
          opt.value = s.game_id;
          opt.textContent = `${date} | ${s.player_names.join(" vs ")} | ${scores} | ${winnerName}`;
          replaySelect.appendChild(opt);
        });
      };

      addGroup("Eval Games", groups.eval);
      addGroup("Human Games", groups.human);
    }
  });

  document.getElementById("menu-close-btn").addEventListener("click", closeMenu);

  document.getElementById("start-game-btn").addEventListener("click", () => {
    const p1 = document.getElementById("player1-select").value;
    const p2 = document.getElementById("player2-select").value;
    const manualFactories = document.getElementById("manual-factories-checkbox").checked;
    closeMenu();
    startNewGame([p1, p2], manualFactories);
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
  _renderInspector();
}

function _applyNewState(newState) {
  gameState = newState;
  selection = null;
  renderLive();
  maybeRunBot();
  if (gameState.is_game_over && gameState.last_game_id) {
    setTimeout(() => loadReplay(gameState.last_game_id), 1500);
  }
}

async function submitMove(source, color, destination) {
  const count = _countInSource(gameState, source, color);

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
  const newState = await res.json();

  if (gameState.in_hypothetical) {
    const playerIndex = gameState.current_player;
    const originalTypes = _hypOriginalPlayerTypes ?? gameState.player_types;
    const isBot = originalTypes[playerIndex] !== "human";
    _recordHypNode({ playerIndex, isBot, source, color, count, destination }, newState);
  }
  _applyNewState(newState);
}

async function requestAgentMove() {
  const res = await fetch(`${API}/agent-move`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json();
    console.error("Agent move failed:", err.detail);
    return;
  }
  const newState = await res.json();

  if (gameState && gameState.in_hypothetical) {
    const playerIndex = gameState.current_player;
    _recordHypNode(
      { playerIndex, isBot: true, source: null, color: null, count: null, destination: null },
      newState
    );
  }
  _applyNewState(newState);
}

async function startNewGame(playerTypes, manualFactories = false) {
  const res = await fetch(`${API}/new-game`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_types: playerTypes, manual_factories: manualFactories }),
  });
  gameState = await res.json();
  selection = null;
  replayRecord = null;
  _hypRoot = null;
  _hypCurrent = null;
  _inspectorReset();
  hasGameInProgress = true;
  renderLive();
  maybeRunBot();
}

async function loadReplay(gameId) {
  replayRecord = await fetchRecord(gameId);
  replayTurnIndex = 0;
  _inspectorReset();
  hasGameInProgress = true;
  renderReplay();
}

// ── Hypothetical mode ──────────────────────────────────────────────────────

async function requestUndo() {
  const res = await fetch(`${API}/undo`, { method: "POST" });
  if (!res.ok) return;
  gameState = await res.json();
  if (_hypCurrent?.parent) _setCurrentNode(_hypCurrent.parent);
  selection = null;
  renderLive();
}

async function enterHypothetical() {
  const res = await fetch(`${API}/hypothetical/enter`, { method: "POST" });
  if (!res.ok) return;
  const prevState = gameState;  // capture before overwrite
  gameState = await res.json();
  selection = null;
  _hypOriginalPlayerTypes = prevState.player_types;
  _hypRoot = _makeNode(null, null, gameState);
  _hypCurrent = _hypRoot;
  renderLive();
}

async function discardHypothetical() {
  const res = await fetch(`${API}/hypothetical/discard`, { method: "POST" });
  if (!res.ok) return;
  gameState = await res.json();
  selection = null;
  _hypRoot = null;
  _hypCurrent = null;
  _hypOriginalPlayerTypes = null;
  renderLive();
  maybeRunBot();
}

async function jumpToNode(node) {
  // Reload the server's in-hyp state from the node's snapshot without
  // exiting hypothetical mode — use the replace endpoint which mutates
  // _game.state directly without touching _hyp_marker or _history.
  const snapshot = _buildSnapshotFromState(node.state);
  const res = await fetch(`${API}/hypothetical/replace-snapshot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(snapshot),
  });
  if (!res.ok) {
    console.error("replace-snapshot failed:", (await res.json()).detail);
    return;
  }
  gameState = await res.json();
  _setCurrentNode(node);
  selection = null;
  renderLive();
}

function _buildSnapshotFromState(state) {
  return {
    current_player: state.current_player,
    factories: state.factories,
    center: state.center,
    boards: state.boards.map(b => ({
      score: b.score,
      wall: b.wall,
      pattern_lines: b.pattern_lines,
      floor_line: b.floor_line,
    })),
  };
}

async function _executeCommit(targetNode) {
  const path = [];
  let node = targetNode;
  while (node && node.move !== null) {
    path.unshift(node);
    node = node.parent;
  }

  const commitRes = await fetch(`${API}/hypothetical/commit`, { method: "POST" });
  if (!commitRes.ok) return;
  gameState = await commitRes.json();
  _hypRoot = null;
  _hypCurrent = null;
  _hypOriginalPlayerTypes = null;
  selection = null;

  for (const pathNode of path) {
    const move = pathNode.move;
    if (move.isBot) {
      const agentRes = await fetch(`${API}/agent-move`, { method: "POST" });
      if (!agentRes.ok) break;
      gameState = await agentRes.json();
      renderLive();
    } else {
      const moveRes = await fetch(`${API}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: move.source, tile: move.color, destination: move.destination }),
      });
      if (!moveRes.ok) break;
      gameState = await moveRes.json();
      renderLive();
    }

    if (!gameState.is_game_over && gameState.player_types[gameState.current_player] !== "human") {
      const agentRes = await fetch(`${API}/agent-move`, { method: "POST" });
      if (!agentRes.ok) break;
      gameState = await agentRes.json();
      renderLive();
    }

    if (gameState.is_game_over) break;
  }

  renderLive();
  maybeRunBot();
}

async function enterHypotheticalFromReplay() {
  const record = replayRecord;
  const turns = record.computed_turns ?? [];
  const isAfterLastMove = replayTurnIndex >= turns.length;

  let boardStates, factories, center, currentPlayer;

  if (isAfterLastMove) {
    boardStates = record.final_boards ?? [];
    factories = [];
    center = [];
    currentPlayer = 0;
  } else {
    const turn = turns[replayTurnIndex];
    boardStates = turn.boards;
    factories = turn.factories;
    center = turn.center;
    currentPlayer = turn.player_index;
  }

  const snapshot = {
    current_player: currentPlayer,
    factories,
    center,
    boards: boardStates.map(bs => ({
      score: bs.score,
      wall: bs.wall,
      pattern_lines: bs.pattern_lines,
      floor_line: bs.floor_line,
    })),
  };

  const res = await fetch(`${API}/hypothetical/from-snapshot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(snapshot),
  });
  if (!res.ok) {
    console.error("from-snapshot failed:", (await res.json()).detail);
    return;
  }
  gameState = await res.json();
  replayRecord = null;
  selection = null;
  _hypOriginalPlayerTypes = ["human", "human"];
  _hypRoot = _makeNode(null, null, gameState);
  _hypCurrent = _hypRoot;
  renderLive();
}

// ── Factory setup API ──────────────────────────────────────────────────────

async function setupPlace(color) {
  const res = await fetch(`${API}/setup-factories/place`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ color }),
  });
  if (!res.ok) { console.error("Setup place failed:", (await res.json()).detail); return; }
  gameState = await res.json();
  renderLive();
}

async function setupRemove(factoryIndex, slotIndex) {
  const res = await fetch(`${API}/setup-factories/remove`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ factory: factoryIndex, slot: slotIndex }),
  });
  if (!res.ok) { console.error("Setup remove failed:", (await res.json()).detail); return; }
  gameState = await res.json();
  renderLive();
}

async function setupRandom() {
  const res = await fetch(`${API}/setup-factories/random`, { method: "POST" });
  if (!res.ok) { console.error("Setup random failed:", (await res.json()).detail); return; }
  gameState = await res.json();
  renderLive();
}

async function setupRestart() {
  const res = await fetch(`${API}/setup-factories/restart`, { method: "POST" });
  if (!res.ok) { console.error("Setup restart failed:", (await res.json()).detail); return; }
  gameState = await res.json();
  renderLive();
}

async function setupCommit() {
  const res = await fetch(`${API}/setup-factories/commit`, { method: "POST" });
  if (!res.ok) { console.error("Setup commit failed:", (await res.json()).detail); return; }
  gameState = await res.json();
  renderLive();
  maybeRunBot();
}

// ── Bot loop ───────────────────────────────────────────────────────────────

function currentPlayerIsBot() {
  if (!gameState) return false;
  return gameState.player_types[gameState.current_player] !== "human";
}

function maybeRunBot() {
  if (!gameState || gameState.is_game_over) return;
  if (gameState.in_factory_setup) return;
  if (gameState.in_hypothetical) return;
  if (!currentPlayerIsBot()) return;
  const isNewRound = lastRound !== null && gameState.round !== lastRound;
  lastRound = gameState.round;
  setTimeout(requestAgentMove, isNewRound ? 2000 : 600);
}

// ── Selection ──────────────────────────────────────────────────────────────

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

// ── Factory setup render ───────────────────────────────────────────────────

function renderFactorySetup(state, app) {
  const filledSlots = state.factories.reduce((sum, f) => sum + f.length, 0);
  const totalSlots  = state.factories.length * 4;
  const allFull     = filledSlots === totalSlots;

  const header = createElement("div", "header");
  header.appendChild(createElement("p", "status-line",
    `Set up factories \u2014 ${filledSlots} / ${totalSlots} tiles placed`));

  const commitBtn = createElement("button", "header-btn", "Commit");
  commitBtn.disabled = !allFull;
  commitBtn.addEventListener("click", setupCommit);
  header.appendChild(commitBtn);

  const randomBtn = createElement("button", "header-btn secondary", "Random");
  randomBtn.addEventListener("click", setupRandom);
  header.appendChild(randomBtn);

  const restartBtn = createElement("button", "header-btn secondary", "Restart");
  restartBtn.disabled = filledSlots === 0;
  restartBtn.addEventListener("click", setupRestart);
  header.appendChild(restartBtn);

  header.appendChild(_makeMenuButton());

  app.appendChild(header);
  app.appendChild(createElement("p", "selection-hint",
    "Click a tile in the Bag row to place it. Click a placed tile to remove it."));
  app.appendChild(renderSources(
    { factories: state.factories, center: state.center,
      bagCounts: state.bag_counts, discardCounts: state.discard_counts },
    { setupMode: true, factoryCursor: state.factory_cursor,
      onBagTileClick: color => setupPlace(color),
      onFactoryRemove: (fi, si) => setupRemove(fi, si) }
  ));

  const boards = createElement("div", "boards");
  state.boards.forEach((board, index) => {
    const agentLabel = PLAYER_OPTIONS.find(o => o.value === state.player_types[index])?.label ?? state.player_types[index];
    const boardLabel = `P${index + 1} ${agentLabel}`;
    boards.appendChild(renderBoard(
      board, index, boardLabel, false, { interactive: false }
    ));
  });
  app.appendChild(boards);
  _renderInspector();
}

// ── Live render ────────────────────────────────────────────────────────────

function renderLive() {
  const state = gameState;
  const app = document.getElementById("app");
  app.innerHTML = "";

  if (state.in_factory_setup) {
    renderFactorySetup(state, app);
    return;
  }

  const isBotVsBot = state.player_types.every(t => t !== "human");
  const header = createElement("div", "header");

  if (state.in_hypothetical) {
    const originallyBot = _hypOriginalPlayerTypes &&
      _hypOriginalPlayerTypes[state.current_player] !== "human";
    header.appendChild(createElement("p", "status-line",
      `Hypothetical \u2014 Player ${state.current_player + 1}\u2019s turn${originallyBot ? " (bot\u2019s move)" : ""}`));
    header.appendChild(createElement("p", "selection-hint",
      selection
        ? `Selected: ${selection.color} from ${selection.source === -1 ? "center" : `factory ${selection.source + 1}`} \u2014 click a pattern line row or the floor`
        : "Exploring a hypothetical line \u2014 moves here are not final"));

    const discardBtn = createElement("button", "header-btn secondary", "Discard");
    discardBtn.addEventListener("click", discardHypothetical);
    header.appendChild(discardBtn);

    const undoBtn = createElement("button", "header-btn secondary", "Undo");
    undoBtn.disabled = !_hypCurrent?.parent;
    undoBtn.addEventListener("click", requestUndo);
    header.appendChild(undoBtn);
  } else {
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

    const undoBtn = createElement("button", "header-btn secondary", "Undo");
    undoBtn.disabled = isBotVsBot || state.is_game_over;
    undoBtn.addEventListener("click", requestUndo);
    header.appendChild(undoBtn);

    const whatIfBtn = createElement("button", "header-btn secondary", "What if?");
    whatIfBtn.disabled = isBotVsBot || state.is_game_over;
    whatIfBtn.addEventListener("click", enterHypothetical);
    header.appendChild(whatIfBtn);
  }

  header.appendChild(_makeMenuButton());
  app.appendChild(header);

  const humanTurn = !currentPlayerIsBot() && !state.is_game_over;
  const inHyp = state.in_hypothetical;

  app.appendChild(renderSources(
    { factories: state.factories, center: state.center,
      bagCounts: state.bag_counts, discardCounts: state.discard_counts },
    (humanTurn || inHyp)
      ? { interactive: true, selection, onTileClick: handleSourceClick }
      : { interactive: false }
  ));

  const boards = createElement("div", "boards");
  state.boards.forEach((board, index) => {
    const isActive = index === state.current_player;
    const canInteract = isActive && (humanTurn || inHyp) && selection !== null;
    const agentLabel = PLAYER_OPTIONS.find(o => o.value === state.player_types[index])?.label ?? state.player_types[index];
    const boardLabel = `P${index + 1} ${agentLabel}`;
    boards.appendChild(renderBoard(board, index, boardLabel, isActive, {
      interactive: canInteract,
      canPlace: row => isLegalDestination(row),
      onRowClick: row => submitMove(selection.source, selection.color, row),
      canPlaceFloor: isLegalDestination(-2),
      onFloorClick: () => submitMove(selection.source, selection.color, -2),
    }));
  });
  app.appendChild(boards);

  if (inHyp && _hypRoot) {
    app.appendChild(renderHypTree(_hypRoot, state.player_types));
  }
  _renderInspector();
}

// ── Replay render ──────────────────────────────────────────────────────────

function renderMoveBanner(turn, record) {
  const banner = createElement("div", "move-banner");
  if (!turn || turn.is_initial || turn.tile === null) {
    banner.appendChild(createElement("span", "move-text", "Start of game \u2014 no move yet."));
    return banner;
  }
  const chip = createElement("span", "move-tile-chip");
  chip.style.background = TILE_COLORS[turn.tile] ?? "#888";
  if (turn.tile === "WHITE") chip.style.border = "1px solid #ccc";
  banner.appendChild(chip);
  const playerName = record.player_names[turn.player_index];
  const src = turn.source === -1 ? "center" : `factory ${turn.source + 1}`;
  const dst = turn.destination === -2 ? "floor" : `row ${turn.destination + 1}`;
  banner.appendChild(createElement("span", "move-text",
    `${playerName} took ${turn.tile} from ${src} \u2192 ${dst}`));
  return banner;
}

function renderReplayMoveList(record, currentIndex) {
  const panel = createElement("div", "replay-move-list-panel");
  panel.appendChild(createElement("div", "replay-move-list-heading", "Move list"));

  const list = createElement("div", "replay-move-list");
  const turns = record.computed_turns ?? [];

  let lastRoundSeen = null;
  let moveNumber = 0;  // 0 = initial state, 1+ = moves

  turns.forEach((turn, index) => {
    // Round header when round changes.
    if (turn.round !== lastRoundSeen) {
      lastRoundSeen = turn.round;
      if (!turn.is_initial) {
        list.appendChild(createElement("div", "replay-round-header",
          `Round ${turn.round}`));
      }
    }

    const item = createElement("div", "replay-move-item");
    if (currentIndex === index) item.classList.add("replay-move-item-current");

    if (turn.is_initial || turn.tile === null) {
      // Start of game entry.
      item.appendChild(createElement("span", "replay-move-item-number", "0)"));
      item.appendChild(createElement("span", "replay-move-item-emoji", "🎲"));
      item.appendChild(createElement("span", "replay-move-item-text", "Start of game"));
      item.appendChild(createElement("span", "replay-move-item-scores",
        turn.grand_totals ? turn.grand_totals.join(" \u2013 ") : "0 \u2013 0"));
    } else {
      moveNumber++;
      const isBot = (record.player_types ?? [])[turn.player_index] !== "human";
      const playerName = record.player_names[turn.player_index];
      const src = turn.source === -1 ? "Ce" : `F${turn.source + 1}`;
      const dst = turn.destination === -2 ? "Floor" : `R${turn.destination + 1}`;

      const chip = document.createElement("span");
      chip.className = "hyp-tile-chip";
      chip.style.background = TILE_COLORS[turn.tile] ?? "#888";
      if (turn.tile === "WHITE") chip.style.border = "1px solid #ccc";

      item.appendChild(createElement("span", "replay-move-item-number", `${moveNumber})`));
      item.appendChild(createElement("span", "replay-move-item-emoji", isBot ? "🤖" : "👤"));
      item.appendChild(createElement("span", "replay-move-item-text", `${playerName} \u00b7`));
      item.appendChild(chip);
      item.appendChild(createElement("span", "replay-move-item-text", `${src} \u2192 ${dst}`));

      const scores = turn.grand_totals && turn.grand_totals.length
        ? turn.grand_totals
        : turn.boards.map(b => b.score);
      item.appendChild(createElement("span", "replay-move-item-scores",
        scores.join(" \u2013 ")));
    }

    item.addEventListener("click", () => { replayTurnIndex = index; _inspectorReset(); renderReplay(); });
    list.appendChild(item);
  });

  // End of game entry.
  if (record.final_scores && record.final_scores.length) {
    list.appendChild(createElement("div", "replay-round-header", "Final"));
    const endItem = createElement("div", "replay-move-item");
    if (currentIndex >= turns.length) endItem.classList.add("replay-move-item-current");
    endItem.appendChild(createElement("span", "replay-move-item-number", ""));
    endItem.appendChild(createElement("span", "replay-move-item-emoji", "🏆"));
    const winnerText = record.winner !== null
      ? `${record.player_names[record.winner]} wins`
      : "Tie";
    endItem.appendChild(createElement("span", "replay-move-item-text",
      `Game over \u2014 ${winnerText}`));
    endItem.appendChild(createElement("span", "replay-move-item-scores",
      record.final_scores.join(" \u2013 ")));
    endItem.addEventListener("click", () => {
      replayTurnIndex = turns.length; _inspectorReset(); renderReplay();
    });
    list.appendChild(endItem);
  }

  panel.appendChild(list);
  return panel;
}

function renderReplay() {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const record = replayRecord;
  const turns = record.computed_turns ?? [];
  const isAfterLastMove = replayTurnIndex >= turns.length;
  const currentTurn = isAfterLastMove ? null : turns[replayTurnIndex];
  const boardStates = isAfterLastMove ? (record.final_boards ?? []) : currentTurn.boards;
  const sourceState = isAfterLastMove
    ? { factories: [], center: [], bag_counts: {}, discard_counts: {} }
    : {
        factories: currentTurn.factories,
        center: currentTurn.center,
        bag_counts: currentTurn.bag_counts,
        discard_counts: currentTurn.discard_counts,
      };

  // Controls.
  const controls = createElement("div", "replay-controls");
  const prevBtn = createElement("button", "replay-btn", "\u25c4 Prev");
  prevBtn.disabled = replayTurnIndex === 0;
  prevBtn.addEventListener("click", () => { replayTurnIndex--; _inspectorReset(); renderReplay(); });
  controls.appendChild(prevBtn);

  const nextBtn = createElement("button", "replay-btn", "Next \u25ba");
  nextBtn.disabled = isAfterLastMove;
  nextBtn.addEventListener("click", () => { replayTurnIndex++; _inspectorReset(); renderReplay(); });
  controls.appendChild(nextBtn);

  controls.appendChild(createElement("span", "turn-counter",
    `Move ${replayTurnIndex} / ${turns.length}`));

  const whatIfBtn = createElement("button", "replay-btn", "What if?");
  whatIfBtn.addEventListener("click", enterHypotheticalFromReplay);
  controls.appendChild(whatIfBtn);

  controls.appendChild(_makeMenuButton("replay-btn"));

  app.appendChild(controls);

  // Move banner.
  app.appendChild(renderMoveBanner(currentTurn, record));

  // Status line.
  if (isAfterLastMove) {
    const winnerText = record.winner !== null
      ? `Game over \u2014 ${record.player_names[record.winner]} wins!`
      : "Game over \u2014 it\u2019s a tie!";
    app.appendChild(createElement("p", "status-line", winnerText));
  } else {
    app.appendChild(createElement("p", "status-line",
      `${record.player_names[currentTurn.player_index]}\u2019s move`));
  }

  // Sources.
  app.appendChild(renderSources(
    { factories: sourceState.factories, center: sourceState.center,
      bagCounts: sourceState.bag_counts, discardCounts: sourceState.discard_counts },
    { interactive: false }
  ));

  // Boards.
  const boards = createElement("div", "boards");
  boardStates.forEach((bs, i) => {
    const board = { ...bs };
    const isActive = !isAfterLastMove && currentTurn && currentTurn.player_index === i;
    boards.appendChild(
      renderBoard(board, i, record.player_names[i], isActive, { interactive: false })
    );
  });
  app.appendChild(boards);

  // Move list.
  app.appendChild(renderReplayMoveList(record, replayTurnIndex));

  setTimeout(() => {
    const current = app.querySelector(".replay-move-item-current");
    if (current) current.scrollIntoView({ block: "center", behavior: "instant" });
  }, 0);
  _renderInspector();
}

initMenu();
openMenu();
document.addEventListener("keydown", (event) => {
  if (!replayRecord) return;
  const turns = replayRecord.computed_turns ?? [];
  if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
    event.preventDefault();
    if (replayTurnIndex > 0) {
      replayTurnIndex--;
      _inspectorReset();
      renderReplay();
    }
  } else if (event.key === "ArrowRight" || event.key === "ArrowDown") {
    event.preventDefault();
    if (replayTurnIndex < turns.length) {
      replayTurnIndex++;
      _inspectorReset();
      renderReplay();
    }
  }
});

