// frontend/render.js
// Shared rendering functions used by both live game and replay.
// No API calls, no game state, no side effects — pure DOM construction.

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

const CENTER_SORT_ORDER = ["FIRST_PLAYER", "BLUE", "YELLOW", "RED", "BLACK", "WHITE"];
const CENTER_SLOTS = 8;

// ── Low-level helpers ──────────────────────────────────────────────────────

function createElement(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

/**
 * Make a tile div.
 * @param {string|null} color  - Tile color name, or null for placeholder.
 * @param {boolean}     faint  - If true, render as a faint wall hint.
 */
function makeTile(color, faint = false) {
  const tile = createElement("div", "tile");
  if (color && !faint) {
    tile.classList.add("tile-placed");
    tile.style.background = TILE_COLORS[color];
    if (color === "WHITE") tile.style.border = "1px solid #ccc";
  } else if (color && faint) {
    tile.style.background = TILE_COLORS[color];
    tile.classList.add("tile-faint");
  }
  // else: plain placeholder, default grey from CSS
  return tile;
}

/**
 * Make a faint info tile for the bag/box panel.
 * Uses a CSS class for opacity so the badge count stays fully opaque.
 */
function makeInfoTile(color, count) {
  const tile = createElement("div", "tile tile-info");
  tile.style.background = TILE_COLORS[color];
  if (color === "WHITE") tile.style.border = "1px solid #ccc";
  tile.appendChild(createElement("div", "stack-badge", `${count}`));
  return tile;
}

function floorPenalty(floorLine) {
  const count = Math.min(floorLine.length, FLOOR_PENALTIES.length);
  return FLOOR_PENALTIES.slice(0, count).reduce((a, b) => a + b, 0);
}

// ── Factories and center ───────────────────────────────────────────────────

/**
 * Render the sources row (factories, center, bag/box).
 *
 * Setup-mode options (all ignored when setupMode is false):
 *   setupMode        {boolean}  - render factories and bag for setup interaction
 *   factoryCursor    {number}   - flat index (0..N*4-1) of the next slot to fill
 *   onBagTileClick   {fn}       - called with (color) when a bag tile is clicked
 *   onFactoryRemove  {fn}       - called with (factoryIndex, slotIndex) when a placed tile is clicked
 */
function renderSources(sources, {
  interactive = false,
  selection = null,
  onTileClick = null,
  setupMode = false,
  factoryCursor = null,
  onBagTileClick = null,
  onFactoryRemove = null,
} = {}) {
  const section = createElement("section", "sources");

  // ── Factories ──
  sources.factories.forEach((factory, factoryIndex) => {
    const display = createElement("div", "factory");
    const grid = createElement("div", "factory-grid");

    for (let slotIndex = 0; slotIndex < 4; slotIndex++) {
      const color = factory[slotIndex] ?? null;
      const flatIndex = factoryIndex * 4 + slotIndex;

      if (setupMode) {
        if (color) {
          // Placed tile — clickable for removal.
          const tile = makeTile(color);
          tile.classList.add("clickable");
          tile.addEventListener("click", () => onFactoryRemove(factoryIndex, slotIndex));
          grid.appendChild(tile);
        } else {
          // Empty placeholder — highlight if it's the cursor slot.
          const tile = makeTile(null);
          if (flatIndex === factoryCursor) {
            tile.classList.add("cursor-slot");
          }
          grid.appendChild(tile);
        }
      } else {
        // Normal live-game interaction.
        const tile = makeTile(color);
        if (interactive && color && color !== "FIRST_PLAYER") {
          tile.classList.add("clickable");
          if (selection?.source === factoryIndex && selection?.color === color) {
            tile.classList.add("selected");
          }
          tile.addEventListener("click", () => onTileClick(factoryIndex, color));
        }
        grid.appendChild(tile);
      }
    }

    display.appendChild(grid);
    section.appendChild(display);
  });

  // ── Center panel ──
  const centerPanel = createElement("div", "source-panel");
  centerPanel.appendChild(createElement("div", "panel-label", "Center"));

  const centerTiles = createElement("div", "center-tiles");

  if (sources.center.length === 0) {
    for (let i = 0; i < CENTER_SLOTS; i++) {
      centerTiles.appendChild(makeTile(null));
    }
  } else if (sources.center.length <= CENTER_SLOTS) {
    sources.center.forEach(color => {
      const tile = makeTile(color);
      if (interactive && color !== "FIRST_PLAYER") {
        tile.classList.add("clickable");
        if (selection?.source === -1 && selection?.color === color) {
          tile.classList.add("selected");
        }
        tile.addEventListener("click", () => onTileClick(-1, color));
      }
      centerTiles.appendChild(tile);
    });
    const remaining = CENTER_SLOTS - sources.center.length;
    for (let i = 0; i < remaining; i++) {
      centerTiles.appendChild(makeTile(null));
    }
  } else {
    // Overflow — sort and stack.
    const sorted = [...sources.center].sort(
      (a, b) => CENTER_SORT_ORDER.indexOf(a) - CENTER_SORT_ORDER.indexOf(b)
    );
    const groups = [];
    for (const color of sorted) {
      if (groups.length && groups[groups.length - 1].color === color) {
        groups[groups.length - 1].count++;
      } else {
        groups.push({ color, count: 1 });
      }
    }
    groups.forEach(({ color, count }) => {
      const tile = makeTile(color);
      if (count > 1) {
        tile.appendChild(createElement("div", "stack-badge", `${count}`));
      }
      if (interactive && color !== "FIRST_PLAYER") {
        tile.classList.add("clickable");
        if (selection?.source === -1 && selection?.color === color) {
          tile.classList.add("selected");
        }
        tile.addEventListener("click", () => onTileClick(-1, color));
      }
      centerTiles.appendChild(tile);
    });
  }

  centerPanel.appendChild(centerTiles);
  section.appendChild(centerPanel);

  // ── Bag / Box panel ──
  if (sources.bagCounts || sources.discardCounts) {
    const COLOR_ORDER = ["BLUE", "YELLOW", "RED", "BLACK", "WHITE"];

    const renderTileRow = (counts, clickable) => {
      const row = createElement("div", "pile-tile-row");
      COLOR_ORDER.forEach(color => {
        const count = counts?.[color] ?? 0;
        let tile;
        if (clickable && count > 0) {
          // Setup mode: full-opacity placed tile, clickable, count badge.
          tile = createElement("div", "tile tile-placed tile-setup-bag");
          tile.style.background = TILE_COLORS[color];
          if (color === "WHITE") tile.style.border = "1px solid #ccc";
          tile.appendChild(createElement("div", "stack-badge", `${count}`));
          tile.classList.add("clickable");
          tile.addEventListener("click", () => onBagTileClick(color));
        } else {
          // Normal mode: faint info tile.
          tile = makeInfoTile(color, count);
        }
        row.appendChild(tile);
      });
      return row;
    };

    const bagDiscardPanel = createElement("div", "bag-discard-panel");

    const bagRow = createElement("div", "pile-labeled-row");
    bagRow.appendChild(createElement("div", "panel-label", "Bag"));
    bagRow.appendChild(renderTileRow(sources.bagCounts, setupMode));
    bagDiscardPanel.appendChild(bagRow);

    const discardRow = createElement("div", "pile-labeled-row");
    discardRow.appendChild(createElement("div", "panel-label", "Box"));
    discardRow.appendChild(renderTileRow(sources.discardCounts, false));
    bagDiscardPanel.appendChild(discardRow);

    section.appendChild(bagDiscardPanel);
  }

  return section;
}

// ── Pattern lines ──────────────────────────────────────────────────────────

function renderPatternLines(patternLines, { interactive = false, canPlace = null, onRowClick = null } = {}) {
  const wrapper = createElement("div", "pattern-lines");

  patternLines.forEach((line, row) => {
    const rowEl = createElement("div", "pattern-row");
    const placeable = interactive && canPlace?.(row);

    if (placeable) {
      rowEl.classList.add("droppable-row");
      rowEl.addEventListener("click", () => onRowClick(row));
    }

    const emptyCount = (row + 1) - line.length;
    for (let i = 0; i < emptyCount; i++) rowEl.appendChild(makeTile(null));
    for (const color of line) rowEl.appendChild(makeTile(color));

    wrapper.appendChild(rowEl);
  });

  return wrapper;
}

// ── Floor line ─────────────────────────────────────────────────────────────

function renderFloorLine(floorLine, { interactive = false, canPlace = false, onFloorClick = null } = {}) {
  const wrapper = createElement("div", "floor-line");
  wrapper.appendChild(createElement("div", "floor-label", "Floor"));

  const tilesRow = createElement("div", "floor-tiles");
  if (interactive && canPlace) {
    tilesRow.classList.add("droppable-row");
    tilesRow.addEventListener("click", onFloorClick);
  }

  FLOOR_PENALTIES.forEach((penalty, i) => {
    const slot = makeTile(floorLine[i] ?? null);
    slot.appendChild(createElement("div", "penalty-label", penalty));
    tilesRow.appendChild(slot);
  });

  wrapper.appendChild(tilesRow);
  return wrapper;
}

// ── Wall ───────────────────────────────────────────────────────────────────

function renderWall(wall, pendingPlacements = [], pendingBonuses = []) {
  const pendingMap = {};
  pendingPlacements.forEach(p => { pendingMap[`${p.row},${p.column}`] = p.placement_points; });

  const rowBonusMap = {};
  pendingBonuses
    .filter(b => b.bonus_type === "row")
    .forEach(b => { rowBonusMap[b.index] = b.bonus_points; });

  const columnBonusMap = {};
  pendingBonuses
    .filter(b => b.bonus_type === "column")
    .forEach(b => { columnBonusMap[b.index] = b.bonus_points; });

  const colorBonusMap = {};
  pendingBonuses
    .filter(b => b.bonus_type === "tile")
    .forEach(b => {
      const colorName = ["BLUE", "YELLOW", "RED", "BLACK", "WHITE"][b.index - 1];
      colorBonusMap[colorName] = b.bonus_points;
    });

  const TILE_SIZE = 36;
  const GAP = 4;
  const TILE_STEP = TILE_SIZE + GAP;

  const wrapper = createElement("div", "wall-wrapper");
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
        tileEl.appendChild(createElement("div", "pending-label", `+${pendingMap[key]}`));
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

  const belowStrip = createElement("div", "below-wall-strip");
  belowStrip.style.width = `${5 * TILE_STEP - GAP}px`;
  belowStrip.style.position = "relative";

  Object.entries(columnBonusMap).forEach(([colIndex, points]) => {
    const label = createElement("span", "bonus-label", `+${points}`);
    label.style.position = "absolute";
    label.style.left = `${colIndex * TILE_STEP + TILE_SIZE / 2}px`;
    label.style.transform = "translateX(-50%)";
    belowStrip.appendChild(label);
  });

  Object.entries(colorBonusMap).forEach(([colorName, points]) => {
    const center = COLOR_BONUS_CENTER[colorName];
    const N = Math.floor(center);
    const px = N * TILE_STEP + TILE_STEP / 2 + TILE_SIZE / 2;
    const label = createElement("span", "bonus-label bonus-color", `+${points}`);
    label.style.position = "absolute";
    label.style.left = `${px}px`;
    label.style.transform = "translateX(-50%)";
    belowStrip.appendChild(label);
  });

  if (Object.keys(columnBonusMap).length > 0 || Object.keys(colorBonusMap).length > 0) {
    wrapper.appendChild(belowStrip);
  }

  return wrapper;
}

// ── Hypothetical tree panel ────────────────────────────────────────────────

/**
 * Render the full hypothetical exploration tree below the boards.
 * @param {object} root         - Root HypNode
 * @param {string[]} playerTypes - e.g. ["human", "greedy"]
 */
function renderHypTree(root, playerTypes) {
  const panel = createElement("div", "hyp-tree-panel");
  panel.appendChild(createElement("div", "hyp-tree-heading", "Exploration tree"));

  const tree = createElement("div", "hyp-tree");
  _renderHypNode(tree, root, playerTypes, 0);
  panel.appendChild(tree);
  return panel;
}

function _moveLabel(move) {
  if (!move || move.source === null) return null;

  const src = move.source === -1 ? "Ce" : `F${move.source + 1}`;
  const dst = move.destination === -2 ? "Fl" : `R${move.destination + 1}`;

  const chip = createElement("span", "hyp-tile-chip");
  chip.style.background = TILE_COLORS[move.color] ?? "#888";
  if (move.color === "WHITE") chip.style.border = "1px solid #555";
  if (move.count > 1) {
    chip.appendChild(createElement("span", "hyp-tile-count", `${move.count}`));
  }

  const label = document.createElement("span");
  label.className = "hyp-move-label";
  label.appendChild(createElement("span", "hyp-src", src));
  label.appendChild(chip);
  label.appendChild(createElement("span", "hyp-dst", `\u2192 ${dst}`));
  return label;
}

function _renderHypNode(container, node, playerTypes, depth) {
  const row = createElement("div", `hyp-node${node.isCurrent ? " hyp-node-current" : ""}`);
  row.style.paddingLeft = `${depth * 20}px`;

  if (node.move === null) {
    // Root node — just scores.
    row.appendChild(createElement("span", "hyp-scores hyp-scores-root",
      node.scores.map((s, i) => `P${i + 1}: ${s}`).join("  \u2013  ")));
  } else {
    const emoji = node.move.isBot ? "🤖" : "👤";
    row.appendChild(createElement("span", "hyp-emoji", emoji));

    const label = _moveLabel(node.move);
    if (label) row.appendChild(label);

    row.appendChild(createElement("span", "hyp-scores",
      node.scores.join(" \u2013 ")));

    // Commit button — triggers execution of this line.
    const commitBtn = createElement("button", "hyp-commit-btn", "Commit");
    commitBtn.addEventListener("click", e => {
      e.stopPropagation();
      // Imported via closure from game.js.
      _executeCommit(node);
    });
    row.appendChild(commitBtn);
  }

  // Clicking the row jumps to that position.
  if (!node.isCurrent) {
    row.style.cursor = "pointer";
    row.addEventListener("click", () => jumpToNode(node));
  }

  container.appendChild(row);

  // Recurse into children.
  node.children.forEach(child => _renderHypNode(container, child, playerTypes, depth + 1));
}

// ── Score bar ──────────────────────────────────────────────────────────────

function renderScoreDisplay(board) {
  const penalty = floorPenalty(board.floor_line);
  const placementPoints = (board.pending_placements ?? []).reduce(
    (sum, p) => sum + p.placement_points, 0
  );
  const bonusPoints = (board.pending_bonuses ?? []).reduce(
    (sum, b) => sum + b.bonus_points, 0
  );
  const grandTotal = board.score + placementPoints + penalty + bonusPoints;

  const bar = createElement("div", "score-bar");
  const add = (text, cls) => bar.appendChild(createElement("span", `score-part ${cls}`, text));

  add(`${board.score}`, "score-carried");
  add(` + ${placementPoints}`, "score-pending");
  add(penalty < 0 ? ` \u2212 ${Math.abs(penalty)}` : ` + 0`, "score-floor");
  add(` + ${bonusPoints}`, "score-bonus");
  add(` = ${grandTotal}`, "score-total");

  return bar;
}

// ── Player board ───────────────────────────────────────────────────────────

function renderBoard(board, index, label, isActive, opts = {}) {
  const wrapper = createElement("div",
    `player-board${isActive ? " active-player" : ""}`);

  const heading = createElement("div", "player-heading");
  heading.appendChild(createElement("span", "player-name", `Player ${index + 1} (${label})`));
  heading.appendChild(renderScoreDisplay(board));
  wrapper.appendChild(heading);

  const middle = createElement("div", "board-middle");
  middle.appendChild(renderPatternLines(board.pattern_lines, {
    interactive: opts.interactive,
    canPlace: opts.canPlace,
    onRowClick: opts.onRowClick,
  }));
  middle.appendChild(createElement("div", "board-divider"));
  middle.appendChild(renderWall(
    board.wall,
    board.pending_placements ?? [],
    board.pending_bonuses ?? [],
  ));
  wrapper.appendChild(middle);
  wrapper.appendChild(renderFloorLine(board.floor_line, {
    interactive: opts.interactive,
    canPlace: opts.canPlaceFloor,
    onFloorClick: opts.onFloorClick,
  }));

  return wrapper;
}

// ── Inspector tree panel ───────────────────────────────────────────────────

/**
 * Render the MCTS inspector panel.
 *
 * @param {object}   snapshot     - Inspector snapshot from the API.
 * @param {object}   opts
 * @param {boolean}  opts.enabled       - Whether the panel is open.
 * @param {boolean}  opts.showPerspective - Whether to show the perspective label.
 * @param {function} opts.onToggle      - Called when toggle button clicked.
 * @param {function} opts.onExtend      - Called when Extend button clicked.
 * @param {function} opts.onNodeToggle  - Called with nodeKey when chevron clicked.
 * @param {Set}      opts.expandedKeys  - Set of node keys currently expanded.
 */
function renderInspectorPanel(snapshot, {
  enabled = false,
  running = false,
  showPerspective = false,
  onToggle = null,
  onStartStop = null,
  onNodeToggle = null,
  expandedKeys = new Set(),
} = {}) {
  const panel = createElement("div", "inspector-panel");

  // ── Header ──
  const header = createElement("div", "inspector-header");
  const titleGroup = createElement("div", "inspector-title-group");
  titleGroup.appendChild(createElement("span", "inspector-title", "Search Tree"));

  if (showPerspective && snapshot?.perspective) {
    titleGroup.appendChild(
      createElement("span", "inspector-perspective", `${snapshot.perspective}'s turn`)
    );
  }
  header.appendChild(titleGroup);

  const toggleBtn = createElement("button", "inspector-toggle-btn",
    enabled ? "Hide" : "Show");
  toggleBtn.addEventListener("click", onToggle);
  header.appendChild(toggleBtn);
  panel.appendChild(header);

  if (!enabled) return panel;

  // ── Status bar ──
  const status = createElement("div", "inspector-status");
  if (!snapshot) {
    status.appendChild(createElement("span", "inspector-status-text", "Loading…"));
  } else if (snapshot.done) {
    status.appendChild(createElement("span", "inspector-status-text inspector-status-done",
      `Stable · ${snapshot.sim_count.toLocaleString()} sims`));
  } else if (!running) {
    status.appendChild(createElement("span", "inspector-status-text",
      `Paused · ${snapshot.sim_count.toLocaleString()} sims`));
  } else {
    const pulse = createElement("span", "inspector-pulse");
    status.appendChild(pulse);
    status.appendChild(createElement("span", "inspector-status-text",
      `Searching · ${snapshot.sim_count.toLocaleString()} sims`));
  }

  const runBtn = createElement("button", "inspector-extend-btn",
    snapshot?.done ? "Done" : (running ? "Pause" : "Start"));
  runBtn.disabled = snapshot?.done ?? false;
  runBtn.addEventListener("click", onStartStop);
  status.appendChild(runBtn);
  panel.appendChild(status);

  const copyBtn = createElement("button", "inspector-extend-btn", "Copy Tree");
  copyBtn.disabled = !snapshot;
  copyBtn.addEventListener("click", () => {
    const text = _inspectorTreeText(snapshot.tree, 0);
    navigator.clipboard.writeText(text).then(() => {
      copyBtn.textContent = "Copied!";
      setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
    });
  });
  status.appendChild(copyBtn);

  if (!snapshot) return panel;

  // ── Tree ──
  const treeEl = createElement("div", "inspector-tree");
  const root = snapshot.tree;

  // Compute global scale: max |value_diff| across all visible nodes.
  const maxAbs = _inspectorMaxAbs(root, expandedKeys, 0);
  const scale = maxAbs > 0 ? maxAbs : 1;

  _renderInspectorNode(treeEl, root, {
    depth: 0,
    scale,
    expandedKeys,
    onNodeToggle,
    isRoot: true,
  });

  panel.appendChild(treeEl);
  return panel;
}

function _inspectorMaxAbs(node, expandedKeys, depth) {
  let max = Math.abs(node.value_diff ?? 0);
  if (depth > 0 && node.children && expandedKeys.has(node.key)) {
    for (const child of node.children) {
      max = Math.max(max, _inspectorMaxAbs(child, expandedKeys, depth + 1));
    }
  } else if (depth === 0 && node.children) {
    for (const child of node.children) {
      max = Math.max(max, Math.abs(child.value_diff ?? 0));
    }
  }
  return max;
}

function _renderInspectorNode(container, node, {
  depth,
  scale,
  expandedKeys,
  onNodeToggle,
  isRoot = false,
}) {
  const hasChildren = node.children && node.children.length > 0;
  const isExpanded = expandedKeys.has(node.key);

  const row = createElement("div", "inspector-row");
  row.style.paddingLeft = `${depth * 18}px`;

  // ── Chevron ──
  const chevron = createElement("span", "inspector-chevron",
    isRoot ? "" : (hasChildren ? (isExpanded ? "▼" : "►") : "·"));
  if (hasChildren && !isRoot) {
    chevron.classList.add("inspector-chevron-clickable");
    chevron.addEventListener("click", () => onNodeToggle(node.key));
  }
  row.appendChild(chevron);

  // ── Move label ──
  const moveLabel = createElement("span", "inspector-move-label",
    isRoot ? "root" : (node.move ?? ""));
  if (isRoot) moveLabel.classList.add("inspector-move-root");
  row.appendChild(moveLabel);

  // ── Bidirectional bar ──
  const barWrap = createElement("div", "inspector-bar-wrap");
  const barLeft = createElement("div", "inspector-bar-left");
  const barZero = createElement("div", "inspector-bar-zero");
  const barRight = createElement("div", "inspector-bar-right");

  const v = node.value_diff ?? 0;
  const pct = Math.min(1, Math.abs(v) / scale) * 50; // max 50% of total width

  if (v >= 0) {
    barLeft.style.flex = "1";
    barRight.style.width = `${pct}%`;
    barRight.style.minWidth = v > 0 ? "2px" : "0";
    barRight.classList.add("inspector-bar-pos");
  } else {
    barLeft.style.width = `${pct}%`;
    barLeft.style.minWidth = "2px";
    barLeft.classList.add("inspector-bar-neg");
    barRight.style.flex = "1";
  }

  barWrap.appendChild(barLeft);
  barWrap.appendChild(barZero);
  barWrap.appendChild(barRight);
  row.appendChild(barWrap);

  // ── Value label ──
  const displayValue = v * 20;
  const mm = (node.minimax_value ?? v) * 20;
  const valueSign = displayValue >= 0 ? "+" : "";
  const mmSign = mm >= 0 ? "+" : "";

  const valueEl = createElement("span", "inspector-value",
    `${valueSign}${displayValue.toFixed(1)}`);
  if (displayValue > 1.0) valueEl.classList.add("inspector-value-pos");
  else if (displayValue < -1.0) valueEl.classList.add("inspector-value-neg");
  row.appendChild(valueEl);

  // Minimax — only show if meaningfully different from avg

  const mmEl = createElement("span", "inspector-minimax",
    `(${mmSign}${mm.toFixed(1)})`);
  if (mm >= 1.0) mmEl.classList.add("inspector-value-pos");
  else if (mm <= -1.0) mmEl.classList.add("inspector-value-neg");
  row.appendChild(mmEl);
  

  // ── Visit count ──
  row.appendChild(createElement("span", "inspector-visits",
    `${node.visits.toLocaleString()}v`));

  container.appendChild(row);

  // ── Children ──
  if (!isRoot && isExpanded && hasChildren) {
    for (const child of node.children) {
      _renderInspectorNode(container, child, {
        depth: depth + 1,
        scale,
        expandedKeys,
        onNodeToggle,
        isRoot: false,
      });
    }
  }

  // Root children always shown
  if (isRoot && hasChildren) {
    for (const child of node.children) {
      _renderInspectorNode(container, child, {
        depth: depth + 1,
        scale,
        expandedKeys,
        onNodeToggle,
        isRoot: false,
      });
    }
  }
}

function _inspectorTreeText(node, depth) {
  if (!node) return "";
  const indent = "  ".repeat(depth);
  const move = node.move ?? "root";
  const avg = (node.value_diff * 20).toFixed(1);
  const mm = ((node.minimax_value ?? node.value_diff) * 20).toFixed(1);
  const avgSign = node.value_diff >= 0 ? "+" : "";
  const mmSign = (node.minimax_value ?? node.value_diff) >= 0 ? "+" : "";
  const visits = node.visits.toLocaleString();
  const boundary = node.is_round_boundary ? " [end]" : "";
  const showMm = Math.abs(
    ((node.minimax_value ?? node.value_diff) - node.value_diff) * 20
  ) > 0.5;
  const mmStr = showMm ? `  mm${mmSign}${mm}pts` : "";
  const line = `${indent}${move}  avg${avgSign}${avg}pts${mmStr}  ${visits}v${boundary}`;

  const children = (node.children ?? [])
    .map(c => _inspectorTreeText(c, depth + 1))
    .join("\n");

  return children ? `${line}\n${children}` : line;
}  