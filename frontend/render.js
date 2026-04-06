// frontend/render.js
// Shared rendering functions used by both game.js and replay.js.
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

// Sort order for center tiles: FIRST_PLAYER first, then row-0 color order.
const CENTER_SORT_ORDER = ["FIRST_PLAYER", "BLUE", "YELLOW", "RED", "BLACK", "WHITE"];

// ── Low-level helpers ──────────────────────────────────────────────────────

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

// ── Factories and center ───────────────────────────────────────────────────

/**
 * Render the factory displays and center pool.
 *
 * @param {object} sources        - { factories, center, bagCounts?, discardCounts? }
 * @param {object} [opts]
 * @param {boolean} [opts.interactive=false]
 * @param {object}  [opts.selection=null]
 * @param {function} [opts.onTileClick]
 */
function renderSources(sources, { interactive = false, selection = null, onTileClick = null } = {}) {
  const section = createElement("section", "sources");

  sources.factories.forEach((factory, index) => {
    const display = createElement("div", "factory");
    const grid = createElement("div", "factory-grid");
    for (let i = 0; i < 4; i++) {
      const color = factory[i] ?? null;
      const tile = makeTile(color);
      if (interactive && color && color !== "FIRST_PLAYER") {
        tile.classList.add("clickable");
        if (selection?.source === index && selection?.color === color) {
          tile.classList.add("selected");
        }
        tile.addEventListener("click", () => onTileClick(index, color));
      }
      grid.appendChild(tile);
    }
    display.appendChild(grid);
    section.appendChild(display);
  });

  const centerPool = createElement("div", "center-pool");
  centerPool.appendChild(createElement("div", "pool-label", "Center"));
  const centerTiles = createElement("div", "center-tiles");

  const sortedCenter = [...sources.center].sort(
    (a, b) => CENTER_SORT_ORDER.indexOf(a) - CENTER_SORT_ORDER.indexOf(b)
  );
  if (sortedCenter.length === 0) {
    centerTiles.appendChild(createElement("span", "empty-label", "empty"));
  } else {
    // Group into runs of the same color.
    const groups = [];
    for (const color of sortedCenter) {
      if (groups.length && groups[groups.length - 1].color === color) {
        groups[groups.length - 1].count++;
      } else {
        groups.push({ color, count: 1 });
      }
    }

    const STACK_OFFSET = 5;   // px per tile behind the front
    const TILE_SIZE   = 36;   // must match .tile in CSS

    groups.forEach(({ color, count }) => {
      const stackHeight = TILE_SIZE + STACK_OFFSET * (count - 1);
      const wrapper = createElement("div", "center-stack");
      wrapper.style.position = "relative";
      wrapper.style.width    = `${TILE_SIZE}px`;
      wrapper.style.height   = `${stackHeight}px`;
      wrapper.style.flexShrink = "0";

      // Render back-to-front so front tile is on top.
      for (let i = count - 1; i >= 0; i--) {
        const tile = makeTile(color);
        tile.style.position = "absolute";
        tile.style.top      = `${i * STACK_OFFSET}px`;
        tile.style.left     = "0";

        if (i === 0) {
          // Front tile: count badge + interaction.
          if (count > 1) {
            const badge = createElement("div", "stack-badge", `${count}`);
            tile.appendChild(badge);
          }
          if (interactive && color !== "FIRST_PLAYER") {
            tile.classList.add("clickable");
            if (selection?.source === -1 && selection?.color === color) {
              tile.classList.add("selected");
            }
            tile.addEventListener("click", () => onTileClick(-1, color));
          }
        }
        wrapper.appendChild(tile);
      }
      centerTiles.appendChild(wrapper);
    });
  }

  centerPool.appendChild(centerTiles);
  section.appendChild(centerPool);

  // Bag and discard counts — shown below center if provided.
  if (sources.bagCounts || sources.discardCounts) {
    const COLOR_ORDER = ["BLUE", "YELLOW", "RED", "BLACK", "WHITE"];
    const renderPile = (label, counts) => {
      const pile = createElement("div", "pile-counts");
      pile.appendChild(createElement("div", "pool-label", label));
      const row = createElement("div", "pile-row");
      COLOR_ORDER.forEach(color => {
        const cell = createElement("div", "pile-cell");
        const chip = createElement("div", "pile-chip");
        chip.style.background = TILE_COLORS[color];
        if (color === "WHITE") chip.style.border = "1px solid #ccc";
        cell.appendChild(chip);
        cell.appendChild(createElement("span", "pile-count", `${counts?.[color] ?? 0}`));
        row.appendChild(cell);
      });
      pile.appendChild(row);
      return pile;
    };
    section.appendChild(renderPile("Bag", sources.bagCounts));
    section.appendChild(renderPile("Discard", sources.discardCounts));
  }

  return section;
}

// ── Pattern lines ──────────────────────────────────────────────────────────

/**
 * @param {string[][]} patternLines
 * @param {object} [opts]
 * @param {boolean}  [opts.interactive=false]
 * @param {function} [opts.canPlace]         - (row) => bool
 * @param {function} [opts.onRowClick]       - (row) => void
 */
function renderPatternLines(patternLines, { interactive = false, canPlace = null, onRowClick = null } = {}) {
  const wrapper = createElement("div", "pattern-lines");

  patternLines.forEach((line, row) => {
    const rowEl = createElement("div", "pattern-row");

    if (interactive && canPlace?.(row)) {
      rowEl.classList.add("clickable-row");
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

/**
 * @param {string[]} floorLine
 * @param {object} [opts]
 * @param {boolean}  [opts.interactive=false]
 * @param {boolean}  [opts.canPlace=false]
 * @param {function} [opts.onFloorClick]
 */
function renderFloorLine(floorLine, { interactive = false, canPlace = false, onFloorClick = null } = {}) {
  const wrapper = createElement("div", "floor-line");
  wrapper.appendChild(createElement("div", "pool-label", "Floor"));

  const tilesRow = createElement("div", "floor-tiles");
  if (interactive && canPlace) {
    tilesRow.classList.add("clickable-row");
    tilesRow.addEventListener("click", onFloorClick);
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

// ── Wall ───────────────────────────────────────────────────────────────────

/**
 * @param {(string|null)[][]} wall
 * @param {object[]} [pendingPlacements=[]]
 * @param {object[]} [pendingBonuses=[]]
 */
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

// ── Score bar ──────────────────────────────────────────────────────────────

/**
 * @param {object} board  - Has score, floor_line, pending_placements, pending_bonuses.
 */
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
  add(penalty < 0 ? ` − ${Math.abs(penalty)}` : ` + 0`, "score-floor");
  add(` + ${bonusPoints}`, "score-bonus");
  add(` = ${grandTotal}`, "score-total");

  return bar;
}

// ── Player board ───────────────────────────────────────────────────────────

/**
 * @param {object} board       - Board state (from API or recording).
 * @param {number} index       - Player index (0 or 1).
 * @param {string} label       - Display name for the player.
 * @param {boolean} isActive   - Whether this is the current player.
 * @param {object} [opts]      - Interactive options forwarded to sub-renderers.
 * @param {boolean} [opts.interactive=false]
 * @param {function} [opts.canPlace]
 * @param {function} [opts.onRowClick]
 * @param {boolean}  [opts.canPlaceFloor=false]
 * @param {function} [opts.onFloorClick]
 */
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